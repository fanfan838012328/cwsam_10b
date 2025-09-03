import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class MoEFFN(nn.Module):
    """MoE-based Feed Forward Network for DINOv3"""
    
    def __init__(
        self,
        embed_dim: int,
        ffn_dim: int,
        num_experts: int = 32,
        k: int = 4,
        noisy_gating: bool = True,
        expert_drop_rate: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        self.expert_drop_rate = expert_drop_rate
        
        # 创建专家网络 - 每个专家是标准的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ffn_dim, bias=True),
                nn.GELU(),
                nn.Dropout(expert_drop_rate),
                nn.Linear(ffn_dim, embed_dim, bias=True),
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        self.gate = nn.Linear(embed_dim, num_experts, bias=False)
        
        # 噪声门控参数
        if noisy_gating:
            self.noise_epsilon = 1e-2
        
        # 负载均衡损失权重
        self.load_balance_loss_weight = 0.01
        
    def _noisy_top_k_gating(self, x: torch.Tensor):
        """噪声Top-K门控机制"""
        batch_size, seq_len, _ = x.shape
        x_flat = x.view(-1, self.embed_dim)
        
        # 基础门控分数
        clean_logits = self.gate(x_flat)  # (B*seq_len, num_experts)
        
        if self.noisy_gating and self.training:
            # 添加噪声以提高多样性
            noise = torch.randn_like(clean_logits) * self.noise_epsilon
            noisy_logits = clean_logits + noise
        else:
            noisy_logits = clean_logits
        
        # Top-K选择
        top_k_logits, top_k_indices = noisy_logits.topk(self.k, dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        # 负载均衡损失计算
        importance = clean_logits.softmax(dim=-1)  # (B*seq_len, num_experts)
        load = importance.mean(dim=0)  # (num_experts,)
        
        return top_k_gates, top_k_indices, load
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch_size, seq_len, embed_dim)
        Returns:
            output: (batch_size, seq_len, embed_dim) - 兼容标准MLP接口
        """
        batch_size, seq_len, embed_dim = x.shape
        x_flat = x.view(-1, embed_dim)  # (B*seq_len, embed_dim)
        
        # 获取门控权重和选择的专家
        gates, indices, load = self._noisy_top_k_gating(x)
        
        # 初始化输出
        output = torch.zeros_like(x_flat)
        
        # 为每个专家处理相应的tokens
        for expert_idx in range(self.num_experts):
            # 找到所有使用当前专家的位置
            expert_mask = (indices == expert_idx)  # (B*seq_len, k)
            
            if not expert_mask.any():
                continue
            
            # 收集使用该专家的所有输入和权重
            inputs_for_expert = []
            weights_for_expert = []
            positions = []
            
            for k_idx in range(self.k):
                mask = expert_mask[:, k_idx]  # (B*seq_len,)
                if mask.any():
                    selected_inputs = x_flat[mask]  # (n_selected, embed_dim)
                    selected_weights = gates[:, k_idx][mask]  # (n_selected,)
                    selected_positions = mask.nonzero().squeeze(1)  # (n_selected,)
                    
                    inputs_for_expert.append(selected_inputs)
                    weights_for_expert.append(selected_weights)
                    positions.extend(selected_positions.tolist())
            
            if not inputs_for_expert:
                continue
            
            # 批量处理该专家的所有输入
            combined_inputs = torch.cat(inputs_for_expert, dim=0)  # (total_selected, embed_dim)
            combined_weights = torch.cat(weights_for_expert, dim=0)  # (total_selected,)
            
            # 专家前向计算
            expert_output = self.experts[expert_idx](combined_inputs)  # (total_selected, embed_dim)
            
            # 应用门控权重
            weighted_output = expert_output * combined_weights.unsqueeze(-1)
            
            # 将结果累加到对应位置
            for i, pos in enumerate(positions):
                output[pos] += weighted_output[i]
        
        # 计算并存储辅助损失（但不返回）
        aux_loss = self._compute_load_balance_loss(load)
        self.last_aux_loss = aux_loss  # 存储以供后续访问
        
        # 恢复形状，只返回输出（兼容标准MLP接口）
        output = output.view(batch_size, seq_len, embed_dim)
        
        return output
    
    def forward_list(self, x_list):
        """DINOv3兼容方法 - 处理输入列表"""
        if isinstance(x_list, list):
            # 处理列表输入
            outputs = []
            for x in x_list:
                outputs.append(self.forward(x))
            return outputs
        else:
            # 单个输入
            return self.forward(x_list)
    
    def _compute_load_balance_loss(self, load: torch.Tensor):
        """计算负载均衡损失，鼓励专家使用均匀分布"""
        # 负载均衡损失：鼓励每个专家的使用率接近1/num_experts
        target_load = 1.0 / self.num_experts
        load_loss = ((load - target_load) ** 2).mean()
        return self.load_balance_loss_weight * load_loss


class DINOV3MoEWrapper(nn.Module):
    """DINOv3 MoE wrapper that replaces FFN layers with MoE"""
    
    def __init__(
        self, 
        dinov3_model,
        moe_layers: Optional[list] = None,  # 指定哪些层使用MoE，None表示后半部分
        num_experts: int = 32,
        k: int = 4,
        expert_capacity_factor: float = 1.5,
        noisy_gating: bool = True,
    ):
        """
        Args:
            dinov3_model: 预训练的DINOv3模型
            moe_layers: 使用MoE的层索引列表，None时默认使用后半部分层
            num_experts: 每层MoE的专家数量
            k: 每次激活的专家数量
            expert_capacity_factor: 专家容量扩展因子
            noisy_gating: 是否使用噪声门控
        """
        super().__init__()
        
        self.backbone = dinov3_model
        
        # 获取模型参数
        self.embed_dim = dinov3_model.embed_dim
        self.depth = dinov3_model.n_blocks
        
        # 确定MoE层
        if moe_layers is None:
            # 默认在后半部分层使用MoE
            start_layer = max(0, self.depth // 2)
            self.moe_layers = list(range(start_layer, self.depth))
        else:
            self.moe_layers = moe_layers
            
        print(f"将在层 {self.moe_layers} 使用MoE，共 {len(self.moe_layers)} 层")
        
        # 创建MoE模块但不立即替换
        self.moe_modules = nn.ModuleDict()
        self.original_mlps = {}  # 保存原始MLP的引用
        
        for layer_idx in self.moe_layers:
            if layer_idx >= len(self.backbone.blocks):
                continue
                
            # 获取原始MLP
            original_block = self.backbone.blocks[layer_idx]
            original_mlp = original_block.mlp
            
            # 推断MLP维度
            if hasattr(original_mlp, 'linear1'):
                mlp_dim = original_mlp.linear1.out_features
            elif hasattr(original_mlp, 'w1'):  # SwiGLU
                mlp_dim = original_mlp.w1.out_features
            elif hasattr(original_mlp, 'fc1'):  # Standard MLP
                mlp_dim = original_mlp.fc1.out_features
            else:
                # 默认使用4倍扩展
                mlp_dim = int(self.embed_dim * 4 * expert_capacity_factor)
            
            # 创建MoE FFN
            moe_ffn = MoEFFN(
                embed_dim=self.embed_dim,
                ffn_dim=mlp_dim,
                num_experts=num_experts,
                k=k,
                noisy_gating=noisy_gating,
            )
            
            # 保存原始MLP和创建的MoE
            self.original_mlps[layer_idx] = original_mlp
            self.moe_modules[str(layer_idx)] = moe_ffn
            
            # 直接替换，不使用包装器
            original_block.mlp = moe_ffn
            
        print(f"成功创建 {len(self.moe_modules)} 个MoE模块")
        
        # 冻结backbone参数（除了被替换的MLP）
        self._freeze_non_moe_parameters()
        
        # 总损失累积
        self.total_aux_loss = 0.0
    
    def _freeze_non_moe_parameters(self):
        """冻结非MoE参数"""
        print("正在设置参数训练状态...")
        
        # 首先冻结所有backbone参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 然后解冻MoE模块参数（已经是backbone的一部分）
        for layer_idx in self.moe_layers:
            if layer_idx < len(self.backbone.blocks):
                block = self.backbone.blocks[layer_idx]
                # 解冻该层的MLP参数（现在是MoE）
                for param in block.mlp.parameters():
                    param.requires_grad = True
        
        # 统计可训练参数
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.backbone.parameters() if not p.requires_grad)
        
        print(f"参数状态设置完成:")
        print(f"  可训练参数: {trainable_params/1e9:.2f}B")
        print(f"  冻结参数: {frozen_params/1e9:.2f}B")
    
    def forward(self, x, masks=None):
        """前向传播"""
        # 重置辅助损失
        self.total_aux_loss = 0.0
        
        # DINOv3前向传播
        if hasattr(self.backbone, 'forward_features'):
            # 如果有forward_features方法
            output = self.backbone.forward_features(x, masks=masks)
        else:
            # 标准前向传播
            output = self.backbone(x, masks=masks)
        
        # 收集MoE辅助损失
        for layer_idx_str, moe_module in self.moe_modules.items():
            if hasattr(moe_module, 'last_aux_loss'):
                self.total_aux_loss += moe_module.last_aux_loss
        
        return output
    
    def get_aux_loss(self):
        """获取累积的辅助损失"""
        return self.total_aux_loss
    
    def get_moe_parameters(self):
        """获取所有MoE参数"""
        moe_params = []
        for module in self.moe_modules.values():
            moe_params.extend(module.parameters())
        return moe_params
    
    def get_parameter_count(self):
        """计算参数数量"""
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        moe_params = sum(p.numel() for p in self.moe_modules.parameters())
        
        # 减去被替换的MLP参数
        replaced_params = 0
        for layer_idx, mlp in self.original_mlps.items():
            replaced_params += sum(p.numel() for p in mlp.parameters())
        
        total_params = backbone_params + moe_params - replaced_params
        
        return {
            'backbone': backbone_params,
            'moe': moe_params,
            'replaced_mlp': replaced_params,
            'total': total_params,
            'total_b': total_params / 1e9  # 以B为单位
        }
    
    def load_dinov3_weights(self, checkpoint_path: str):
        """加载DINOv3预训练权重 - 改进版避免递归"""
        print(f"正在加载预训练权重从: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 提取模型权重
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        print(f"原始state_dict包含 {len(state_dict)} 个键")
        
        # 手动加载权重，避免递归问题
        loaded_count = 0
        skipped_count = 0
        
        for name, param in self.backbone.named_parameters():
            # 跳过被MoE替换的MLP层
            is_moe_layer = False
            for layer_idx in self.moe_layers:
                if f'blocks.{layer_idx}.mlp' in name:
                    is_moe_layer = True
                    skipped_count += 1
                    break
            
            if not is_moe_layer and name in state_dict:
                try:
                    param.data.copy_(state_dict[name])
                    loaded_count += 1
                except Exception as e:
                    print(f"加载权重失败 {name}: {e}")
                    skipped_count += 1
            elif not is_moe_layer:
                print(f"权重不存在于预训练模型中: {name}")
                skipped_count += 1
        
        print(f"加载权重完成")
        print(f"成功加载: {loaded_count} 个参数")
        print(f"跳过/失败: {skipped_count} 个参数 (主要是被MoE替换的MLP层)")
        
        return [], []  # 返回空列表，保持接口兼容


def create_dinov3_10b_moe(
    dinov3_checkpoint_path: str,
    num_experts: int = 32,
    k: int = 4,
    moe_layers: Optional[list] = None,
    expert_capacity_factor: float = 1.5,
):
    """创建10B参数的DINOv3-MoE模型"""
    
    print("正在创建DINOv3-MoE模型...")
    
    # 先创建基础模型结构（不加载权重）
    dinov3_model = torch.hub.load('dinov3-main', 'dinov3_vit7b16', source='local', pretrained=False)
    
    # 创建MoE包装器
    moe_model = DINOV3MoEWrapper(
        dinov3_model=dinov3_model,
        moe_layers=moe_layers,
        num_experts=num_experts,
        k=k,
        expert_capacity_factor=expert_capacity_factor,
        noisy_gating=True,
    )
    
    # 安全地加载预训练权重
    try:
        moe_model.load_dinov3_weights(dinov3_checkpoint_path)
        print("预训练权重加载完成")
    except Exception as e:
        print(f"权重加载失败，将使用随机初始化: {e}")
        print("注意: 这将需要从头开始训练")
    
    # 打印参数统计
    param_stats = moe_model.get_parameter_count()
    print(f"\n模型参数统计:")
    print(f"  DINOv3 Backbone: {param_stats['backbone']/1e9:.2f}B")
    print(f"  MoE Modules: {param_stats['moe']/1e9:.2f}B") 
    print(f"  被替换的MLP: {param_stats['replaced_mlp']/1e9:.2f}B")
    print(f"  总参数量: {param_stats['total_b']:.2f}B")
    
    return moe_model