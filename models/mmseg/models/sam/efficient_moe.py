"""
高效MoE架构实现
结合Switch Transformer和Expert Choice的优势
专为10B参数ClassWise-SAM优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Type
from torch.utils.checkpoint import checkpoint


class EfficientMoEMLPBlock(nn.Module):
    """
    高效MoE实现，结合以下优化：
    1. Switch Transformer的简单路由
    2. Expert Choice的负载均衡
    3. 专家分片减少内存占用
    4. 异步专家计算
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        num_experts: int = 48,  # 3组 × 16专家
        expert_capacity_factor: float = 1.5,
        use_expert_choice: bool = True,
        use_sharding: bool = True,
        num_shards: int = 4,
        dropout_rate: float = 0.1,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.expert_capacity_factor = expert_capacity_factor
        self.use_expert_choice = use_expert_choice
        self.use_sharding = use_sharding
        self.num_shards = num_shards if use_sharding else 1
        self.use_checkpoint = use_checkpoint
        
        # 简单路由器
        self.router = nn.Linear(embedding_dim, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        
        # 创建专家网络
        if use_sharding:
            self._create_sharded_experts(act, dropout_rate)
        else:
            self._create_standard_experts(act, dropout_rate)
        
        # 负载均衡损失权重
        self.load_balance_loss_weight = 0.01
        
    def _create_sharded_experts(self, act, dropout_rate):
        """创建分片专家网络"""
        self.expert_shards = nn.ModuleList()
        shard_mlp_dim = self.mlp_dim // self.num_shards
        shard_embed_dim = self.embedding_dim // self.num_shards
        
        for _ in range(self.num_experts):
            expert_shards = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.embedding_dim, shard_mlp_dim),
                    act(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(shard_mlp_dim, shard_embed_dim)
                ) for _ in range(self.num_shards)
            ])
            self.expert_shards.append(expert_shards)
    
    def _create_standard_experts(self, act, dropout_rate):
        """创建标准专家网络"""
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.embedding_dim, self.mlp_dim),
                act(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.mlp_dim, self.embedding_dim)
            ) for _ in range(self.num_experts)
        ])
    
    def _compute_load_balance_loss(self, router_probs, expert_indices):
        """计算负载均衡损失"""
        # 专家使用频率
        expert_usage = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_usage[i] = (expert_indices == i).float().mean()
        
        # 理想情况下每个专家使用频率应该是 1/num_experts
        target_usage = 1.0 / self.num_experts
        load_balance_loss = torch.var(expert_usage) / (target_usage ** 2)
        
        return load_balance_loss
    
    def _expert_choice_routing(self, x_flat, router_logits):
        """Expert Choice路由：专家选择token"""
        total_tokens = x_flat.shape[0]
        expert_capacity = int(self.expert_capacity_factor * total_tokens / self.num_experts)
        
        output = torch.zeros_like(x_flat)
        expert_indices = []
        
        # 每个专家选择最适合的token
        for expert_id in range(self.num_experts):
            expert_scores = router_logits[:, expert_id]
            
            # 选择top-k个token
            if expert_capacity < total_tokens:
                top_tokens = torch.topk(expert_scores, expert_capacity, dim=0)
                selected_indices = top_tokens.indices
                selected_weights = F.softmax(top_tokens.values, dim=0)
            else:
                selected_indices = torch.arange(total_tokens, device=x_flat.device)
                selected_weights = F.softmax(expert_scores, dim=0)
            
            if len(selected_indices) == 0:
                continue
            
            expert_indices.extend([expert_id] * len(selected_indices))
            
            # 处理选中的token
            expert_input = x_flat[selected_indices]
            
            if self.use_sharding:
                expert_output = self._forward_sharded_expert(expert_id, expert_input)
            else:
                if self.use_checkpoint and self.training:
                    expert_output = checkpoint(self.experts[expert_id], expert_input, use_reentrant=False)
                else:
                    expert_output = self.experts[expert_id](expert_input)
            
            # 应用权重并累加
            expert_output *= selected_weights.unsqueeze(-1)
            output[selected_indices] += expert_output
        
        return output, torch.tensor(expert_indices, device=x_flat.device)
    
    def _switch_routing(self, x_flat, router_logits):
        """Switch Transformer路由：每个token选择1个专家"""
        router_probs = F.softmax(router_logits, dim=-1)
        expert_indices = torch.argmax(router_probs, dim=-1)
        expert_weights = torch.max(router_probs, dim=-1)[0]
        
        output = torch.zeros_like(x_flat)
        
        # 批量处理每个专家
        for expert_id in range(self.num_experts):
            expert_mask = expert_indices == expert_id
            if not expert_mask.any():
                continue
            
            expert_input = x_flat[expert_mask]
            weights = expert_weights[expert_mask]
            
            if self.use_sharding:
                expert_output = self._forward_sharded_expert(expert_id, expert_input)
            else:
                if self.use_checkpoint and self.training:
                    expert_output = checkpoint(self.experts[expert_id], expert_input, use_reentrant=False)
                else:
                    expert_output = self.experts[expert_id](expert_input)
            
            expert_output *= weights.unsqueeze(-1)
            output[expert_mask] = expert_output
        
        return output, expert_indices
    
    def _forward_sharded_expert(self, expert_id, expert_input):
        """前向传播分片专家"""
        shard_outputs = []
        
        # 并行处理所有分片
        for shard in self.expert_shards[expert_id]:
            if self.use_checkpoint and self.training:
                shard_output = checkpoint(shard, expert_input, use_reentrant=False)
            else:
                shard_output = shard(expert_input)
            shard_outputs.append(shard_output)
        
        # 连接分片输出
        return torch.cat(shard_outputs, dim=-1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.embedding_dim)
        
        # 计算路由分数
        router_logits = self.router(x_flat)
        
        # 选择路由策略
        if self.use_expert_choice:
            output, expert_indices = self._expert_choice_routing(x_flat, router_logits)
        else:
            output, expert_indices = self._switch_routing(x_flat, router_logits)
        
        # 计算负载均衡损失（仅在训练时）
        if self.training:
            router_probs = F.softmax(router_logits, dim=-1)
            load_balance_loss = self._compute_load_balance_loss(router_probs, expert_indices)
            # 将损失存储为模块属性，可以在训练循环中访问
            self.load_balance_loss = load_balance_loss * self.load_balance_loss_weight
        
        return output.reshape(original_shape)
    
    @classmethod
    def create_from_1_5b_experts(cls, embedding_dim, mlp_dim, original_experts_dict, **kwargs):
        """
        从1.5B模型的专家权重创建新的高效MoE
        
        Args:
            embedding_dim: 嵌入维度
            mlp_dim: MLP隐藏层维度
            original_experts_dict: 原始1.5B模型的专家权重字典
            **kwargs: 其他参数
        """
        # 创建新的MoE模块
        moe_block = cls(embedding_dim=embedding_dim, mlp_dim=mlp_dim, **kwargs)
        
        # 加载原始专家权重
        moe_block.load_1_5b_expert_weights(original_experts_dict)
        
        return moe_block
    
    def load_1_5b_expert_weights(self, original_experts_dict):
        """
        加载1.5B模型的专家权重到新架构
        策略：前16个专家完全复制1.5B权重并冻结，其他专家使用复制+噪声可训练
        """
        if not original_experts_dict:
            print("警告: 没有找到原始专家权重")
            return
        
        # 解析原始专家权重
        original_expert_weights = self._parse_original_expert_weights(original_experts_dict)
        
        if not original_expert_weights:
            print("警告: 无法解析原始专家权重")
            return
        
        # 只在第一次加载时打印，减少重复日志
        if not hasattr(self, '_logged_expert_weights'):
            print(f"找到 {len(original_expert_weights)} 个原始专家权重")
            self._logged_expert_weights = True
        
        with torch.no_grad():
            if self.use_sharding:
                self._load_weights_to_sharded_experts(original_expert_weights)
            else:
                self._load_weights_to_standard_experts(original_expert_weights)
        
        # 权重加载完成，但不在这里冻结专家
        # 冻结逻辑将在分层MoE级别处理
        
        # 只在第一次加载完成时打印，减少重复日志
        if not hasattr(self, '_logged_completion'):
            print("1.5B专家权重加载完成")
            self._logged_completion = True
    
    def _parse_original_expert_weights(self, checkpoint_dict):
        """解析原始checkpoint中的专家权重"""
        expert_weights = {}
        
        for key, value in checkpoint_dict.items():
            if 'experts.' in key and 'image_encoder' in key:
                # 解析专家索引，例如 'image_encoder.blocks.28.mlp.experts.0.0.weight'
                parts = key.split('.')
                expert_idx = None
                for i, part in enumerate(parts):
                    if part == 'experts' and i + 1 < len(parts):
                        try:
                            expert_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                
                if expert_idx is not None and 0 <= expert_idx < 16:
                    # 重构key，移除layer信息，只保留专家相对路径
                    expert_key = '.'.join(parts[parts.index('experts'):])
                    if expert_idx not in expert_weights:
                        expert_weights[expert_idx] = {}
                    
                    # 移除专家索引，获取权重路径
                    weight_path = '.'.join(parts[parts.index('experts') + 2:])
                    expert_weights[expert_idx][weight_path] = value
        
        return expert_weights
    
    def _load_weights_to_sharded_experts(self, original_expert_weights):
        """加载权重到分片专家"""
        num_original_experts = len(original_expert_weights)
        experts_per_shard = self.num_experts // self.num_shards if self.num_shards > 0 else self.num_experts
        
        for expert_id in range(min(self.num_experts, num_original_experts)):
            if expert_id not in original_expert_weights:
                continue
            
            original_weights = original_expert_weights[expert_id]
            expert_shards = self.expert_shards[expert_id]
            
            # 为每个分片分配原始权重的一部分
            for shard_id, shard in enumerate(expert_shards):
                self._load_shard_weights(shard, original_weights, shard_id)
        
        # 为剩余的专家使用复制+噪声策略
        if self.num_experts > num_original_experts:
            self._initialize_remaining_experts(original_expert_weights, num_original_experts)
    
    def _load_weights_to_standard_experts(self, original_expert_weights):
        """加载权重到标准专家"""
        num_original_experts = len(original_expert_weights)
        
        for expert_id in range(min(self.num_experts, num_original_experts)):
            if expert_id not in original_expert_weights:
                continue
            
            original_weights = original_expert_weights[expert_id]
            expert = self.experts[expert_id]
            
            # 直接映射权重
            self._map_weights_to_expert(expert, original_weights)
        
        # 为剩余的专家使用复制+噪声策略
        if self.num_experts > num_original_experts:
            self._initialize_remaining_standard_experts(original_expert_weights, num_original_experts)
    
    def _load_shard_weights(self, shard, original_weights, shard_id):
        """将原始权重映射到分片"""
        shard_dim = self.mlp_dim // self.num_shards
        embed_shard_dim = self.embedding_dim // self.num_shards
        
        for name, param in shard.named_parameters():
            if '0.weight' in name and param.shape[0] == shard_dim:  # 第一层权重
                if '0.weight' in original_weights:
                    orig_weight = original_weights['0.weight']
                    start_idx = shard_id * shard_dim
                    end_idx = start_idx + shard_dim
                    if end_idx <= orig_weight.shape[0]:
                        param.data.copy_(orig_weight[start_idx:end_idx])
                    
            elif '0.bias' in name and param.shape[0] == shard_dim:  # 第一层偏置
                if '0.bias' in original_weights:
                    orig_bias = original_weights['0.bias']
                    start_idx = shard_id * shard_dim
                    end_idx = start_idx + shard_dim
                    if end_idx <= orig_bias.shape[0]:
                        param.data.copy_(orig_bias[start_idx:end_idx])
                        
            elif '2.weight' in name and param.shape[1] == shard_dim:  # 第二层权重
                if '2.weight' in original_weights:
                    orig_weight = original_weights['2.weight']
                    embed_start_idx = shard_id * embed_shard_dim
                    embed_end_idx = embed_start_idx + embed_shard_dim
                    mlp_start_idx = shard_id * shard_dim
                    mlp_end_idx = mlp_start_idx + shard_dim
                    
                    if (embed_end_idx <= orig_weight.shape[0] and 
                        mlp_end_idx <= orig_weight.shape[1]):
                        param.data.copy_(orig_weight[embed_start_idx:embed_end_idx, 
                                                   mlp_start_idx:mlp_end_idx])
                        
            elif '2.bias' in name and param.shape[0] == embed_shard_dim:  # 第二层偏置
                if '2.bias' in original_weights:
                    orig_bias = original_weights['2.bias']
                    start_idx = shard_id * embed_shard_dim
                    end_idx = start_idx + embed_shard_dim
                    if end_idx <= orig_bias.shape[0]:
                        param.data.copy_(orig_bias[start_idx:end_idx])
    
    def _map_weights_to_expert(self, expert, original_weights):
        """将原始权重映射到专家网络"""
        for name, param in expert.named_parameters():
            if name in original_weights:
                orig_param = original_weights[name]
                if param.shape == orig_param.shape:
                    param.data.copy_(orig_param)
                else:
                    print(f"警告: 权重形状不匹配 {name}: {param.shape} vs {orig_param.shape}")
    
    def _initialize_remaining_experts(self, original_expert_weights, start_idx):
        """初始化剩余的分片专家"""
        num_original = len(original_expert_weights)
        
        for expert_id in range(start_idx, self.num_experts):
            # 选择一个原始专家作为模板
            template_idx = expert_id % num_original
            if template_idx in original_expert_weights:
                template_weights = original_expert_weights[template_idx]
                expert_shards = self.expert_shards[expert_id]
                
                # 添加噪声的强度随专家索引增加
                noise_std = 0.02 * (1 + (expert_id - start_idx) / (self.num_experts - start_idx))
                
                for shard_id, shard in enumerate(expert_shards):
                    self._load_shard_weights(shard, template_weights, shard_id)
                    # 添加噪声
                    with torch.no_grad():
                        for param in shard.parameters():
                            noise = torch.randn_like(param) * noise_std
                            param.data.add_(noise)
    
    def _initialize_remaining_standard_experts(self, original_expert_weights, start_idx):
        """初始化剩余的标准专家"""
        num_original = len(original_expert_weights)
        
        for expert_id in range(start_idx, self.num_experts):
            # 选择一个原始专家作为模板
            template_idx = expert_id % num_original
            if template_idx in original_expert_weights:
                template_weights = original_expert_weights[template_idx]
                expert = self.experts[expert_id]
                
                # 加载模板权重
                self._map_weights_to_expert(expert, template_weights)
                
                # 添加噪声
                noise_std = 0.02 * (1 + (expert_id - start_idx) / (self.num_experts - start_idx))
                with torch.no_grad():
                    for param in expert.parameters():
                        noise = torch.randn_like(param) * noise_std
                        param.data.add_(noise)
    
    def _freeze_original_experts(self):
        """冻结前16个专家的参数（完全复制的1.5B权重）"""
        frozen_count = 0
        
        if self.use_sharding:
            # 冻结前16个分片专家
            for expert_id in range(min(16, self.num_experts)):
                if expert_id < len(self.expert_shards):
                    for shard in self.expert_shards[expert_id]:
                        for param in shard.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()
        else:
            # 冻结前16个标准专家
            for expert_id in range(min(16, self.num_experts)):
                if expert_id < len(self.experts):
                    for param in self.experts[expert_id].parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
        
        # 只在第一次冻结时打印，减少重复日志
        if not hasattr(self, '_logged_frozen_params'):
            print(f"  已冻结 {frozen_count:,} 个原始专家参数")
            self._logged_frozen_params = True
        return frozen_count


class OptimizedHierarchicalMoEMLPBlock(nn.Module):
    """
    优化的分层MoE：
    1. 减少专家组数量但保持总参数量
    2. 使用Expert Choice路由
    3. 异步专家计算
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        num_expert_groups: int = 3,  # 减少到3组
        experts_per_group: int = 16,
        k_groups: int = 1,  # 只选择1个组
        k_experts: int = 4,
        expert_capacity_factor: float = 1.5,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.num_expert_groups = num_expert_groups
        self.experts_per_group = experts_per_group
        self.k_groups = k_groups
        self.k_experts = k_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.use_checkpoint = use_checkpoint
        
        # 第一级路由：选择专家组
        self.group_router = nn.Linear(embedding_dim, num_expert_groups, bias=False)
        
        # 专家组：使用高效MoE
        self.expert_groups = nn.ModuleList([
            EfficientMoEMLPBlock(
                embedding_dim=embedding_dim,
                mlp_dim=mlp_dim,
                act=act,
                num_experts=experts_per_group,
                expert_capacity_factor=expert_capacity_factor,
                use_expert_choice=True,  # 使用Expert Choice
                use_sharding=True,       # 启用分片
                use_checkpoint=use_checkpoint,
            ) for _ in range(num_expert_groups)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        
        # 组级路由：选择最优的专家组
        group_logits = self.group_router(x_flat)
        
        if self.k_groups == 1:
            # 如果只选择1个组，使用argmax
            group_indices = torch.argmax(group_logits, dim=-1)
            group_weights = torch.ones_like(group_indices, dtype=torch.float)
        else:
            # 多组选择
            top_k_logits, group_indices = group_logits.topk(self.k_groups, dim=-1)
            group_weights = F.softmax(top_k_logits, dim=-1)
        
        output = torch.zeros_like(x_flat)
        total_load_balance_loss = 0.0
        
        # 处理每个选中的专家组
        for k_idx in range(self.k_groups):
            for group_id in range(self.num_expert_groups):
                if self.k_groups == 1:
                    mask = group_indices == group_id
                    weights = group_weights[mask]
                else:
                    mask = group_indices[:, k_idx] == group_id
                    weights = group_weights[:, k_idx][mask]
                
                if not mask.any():
                    continue
                
                # 将token发送到对应的专家组
                group_input = x_flat[mask]
                group_output = self.expert_groups[group_id](group_input)
                
                # 累加负载均衡损失
                if hasattr(self.expert_groups[group_id], 'load_balance_loss'):
                    total_load_balance_loss += self.expert_groups[group_id].load_balance_loss
                
                # 应用组权重
                if self.k_groups > 1:
                    group_output *= weights.unsqueeze(-1)
                
                output[mask] += group_output
        
        # 存储总的负载均衡损失
        if self.training:
            self.load_balance_loss = total_load_balance_loss / self.num_expert_groups
        
        return output.reshape(original_shape)
    
    @classmethod
    def create_from_1_5b_experts(cls, embedding_dim, mlp_dim, original_experts_dict, **kwargs):
        """
        从1.5B模型的专家权重创建新的优化分层MoE
        """
        # 创建新的MoE模块
        moe_block = cls(embedding_dim=embedding_dim, mlp_dim=mlp_dim, **kwargs)
        
        # 加载原始专家权重到各个专家组
        moe_block.load_1_5b_expert_weights(original_experts_dict)
        
        return moe_block
    
    def load_1_5b_expert_weights(self, original_experts_dict):
        """
        加载1.5B模型的专家权重到分层MoE架构
        策略：第一个专家组的前16个专家完全复制1.5B权重并冻结
        """
        if not original_experts_dict:
            print("警告: 没有找到原始专家权重")
            return
        
        print(f"开始加载1.5B专家权重到{self.num_expert_groups}个专家组...")
        
        total_frozen = 0
        # 为每个专家组加载权重
        for group_idx in range(self.num_expert_groups):
            expert_group = self.expert_groups[group_idx]
            
            # 只在第一次加载时打印进度，减少重复日志
            if hasattr(expert_group, 'load_1_5b_expert_weights'):
                if not hasattr(self, '_logged_group_progress'):
                    print(f"正在加载专家组 {group_idx + 1}/{self.num_expert_groups}")
                    self._logged_group_progress = True
                expert_group.load_1_5b_expert_weights(original_experts_dict)
                
                # 处理专家组的训练状态
                if group_idx == 0:
                    # 第一个专家组：完全冻结（不训练）
                    frozen_count = self._freeze_entire_expert_group(expert_group)
                    total_frozen += frozen_count
                    # 只在第一次冻结时打印，减少重复日志
                    if not hasattr(self, '_logged_first_group_freeze'):
                        print(f"  第一个专家组已完全冻结 {frozen_count:,} 个参数")
                        self._logged_first_group_freeze = True
                else:
                    # 其他专家组：先启用所有参数训练，然后冻结前16个专家
                    self._enable_expert_group_training(expert_group)
                    # 冻结前16个专家（保持1.5B权重）
                    frozen_count = self._freeze_expert_group_original_experts(expert_group)
                    total_frozen += frozen_count
            else:
                print(f"警告: 专家组 {group_idx} 不支持权重加载")
        
        # 只在第一次加载完成时打印总结，减少重复日志
        if not hasattr(self, '_logged_hierarchical_completion'):
            print(f"分层MoE权重加载完成，总共冻结 {total_frozen:,} 个参数")
            self._logged_hierarchical_completion = True
    
    def _freeze_first_group_original_experts(self, expert_group):
        """冻结第一个专家组中前16个专家的参数"""
        frozen_count = 0
        
        if hasattr(expert_group, 'use_sharding') and expert_group.use_sharding:
            # 分片专家
            for expert_id in range(min(16, expert_group.num_experts)):
                if expert_id < len(expert_group.expert_shards):
                    for shard in expert_group.expert_shards[expert_id]:
                        for param in shard.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()
        else:
            # 标准专家
            for expert_id in range(min(16, expert_group.num_experts)):
                if hasattr(expert_group, 'experts') and expert_id < len(expert_group.experts):
                    for param in expert_group.experts[expert_id].parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
        
        return frozen_count
    
    def _freeze_entire_expert_group(self, expert_group):
        """完全冻结一个专家组的所有参数"""
        frozen_count = 0
        
        # 冻结路由器参数
        if hasattr(expert_group, 'router'):
            for param in expert_group.router.parameters():
                param.requires_grad = False
                frozen_count += param.numel()
        
        # 冻结所有专家参数
        if hasattr(expert_group, 'use_sharding') and expert_group.use_sharding:
            # 分片专家
            for expert_shard_list in expert_group.expert_shards:
                for shard in expert_shard_list:
                    for param in shard.parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
        else:
            # 标准专家
            if hasattr(expert_group, 'experts'):
                for expert in expert_group.experts:
                    for param in expert.parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
        
        return frozen_count
    
    def _enable_expert_group_training(self, expert_group):
        """启用专家组的训练（确保参数可训练）"""
        # 启用路由器参数训练
        if hasattr(expert_group, 'router'):
            for param in expert_group.router.parameters():
                param.requires_grad = True
        
        # 启用所有专家参数训练
        if hasattr(expert_group, 'use_sharding') and expert_group.use_sharding:
            # 分片专家
            for expert_shard_list in expert_group.expert_shards:
                for shard in expert_shard_list:
                    for param in shard.parameters():
                        param.requires_grad = True
        else:
            # 标准专家
            if hasattr(expert_group, 'experts'):
                for expert in expert_group.experts:
                    for param in expert.parameters():
                        param.requires_grad = True
    
    def _freeze_expert_group_original_experts(self, expert_group):
        """冻结专家组中前16个专家的参数（保持1.5B权重）"""
        frozen_count = 0
        
        if hasattr(expert_group, 'use_sharding') and expert_group.use_sharding:
            # 冻结前16个分片专家
            for expert_id in range(min(16, expert_group.num_experts)):
                if expert_id < len(expert_group.expert_shards):
                    for shard in expert_group.expert_shards[expert_id]:
                        for param in shard.parameters():
                            param.requires_grad = False
                            frozen_count += param.numel()
        else:
            # 冻结前16个标准专家
            for expert_id in range(min(16, expert_group.num_experts)):
                if hasattr(expert_group, 'experts') and expert_id < len(expert_group.experts):
                    for param in expert_group.experts[expert_id].parameters():
                        param.requires_grad = False
                        frozen_count += param.numel()
        
        return frozen_count