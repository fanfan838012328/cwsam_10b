# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Type
import loralib as lora
from torch.utils.checkpoint import checkpoint


class MoEMLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        num_experts: int = 64,    # 专家数量
        k: int = 3,              # 每次选择的专家数量
        noisy_gating: bool = True,  # 是否使用噪声门控
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_experts = num_experts
        self.k = k
        self.noisy_gating = noisy_gating
        
        # 创建专家网络，每个专家是一个标准的FFN
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                act(),
                nn.Linear(mlp_dim, embedding_dim)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络，决定使用哪些专家
        self.gate = nn.Linear(embedding_dim, num_experts)
        
        if self.noisy_gating:
            self.noise_epsilon = 1e-2
    
    def _noisy_top_k_gating(self, x):
        """使用带噪音的Top-K门控机制"""
        clean_logits = self.gate(x)
        
        if self.noisy_gating and self.training:
            # 添加门控噪声以增加随机性
            noise = torch.randn_like(clean_logits) * self.noise_epsilon
            logits = clean_logits + noise
        else:
            logits = clean_logits
            
        # 计算top-k门控权重
        top_k_logits, top_k_indices = logits.topk(min(self.k, self.num_experts), dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        return top_k_gates, top_k_indices, clean_logits
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 使用更高效的批处理方式处理专家计算
        batch_size = x.shape[0]
        x_flat = x.reshape(-1, self.embedding_dim)
        
        # 获取门控值和专家索引
        gates, indices, _ = self._noisy_top_k_gating(x_flat)
        
        # 使用稀疏操作而不是循环所有专家
        final_output = torch.zeros_like(x_flat)
        
        # 仅处理选中的专家
        for expert_idx in range(self.num_experts):
            # 找出使用当前专家的所有样本位置
            masks = (indices == expert_idx)
            if not masks.any():
                continue
            
            # 对每个专家批量处理相关位置的输入
            expert_inputs = []
            expert_gates = []
            
            for k in range(self.k):
                mask = masks[:, k]
                if mask.any():
                    expert_inputs.append(x_flat[mask])
                    expert_gates.append(gates[:, k][mask])
            
            if not expert_inputs:
                continue
                
            # 组合为一个批次处理
            combined_input = torch.cat(expert_inputs, dim=0)
            combined_gates = torch.cat(expert_gates, dim=0)
            
            # 批量前向计算
            combined_output = self.experts[expert_idx](combined_input)
            
            # 应用门控权重
            combined_output = combined_output * combined_gates.unsqueeze(-1)
            
            # 将结果放回正确位置
            offset = 0
            for k in range(self.k):
                mask = masks[:, k]
                if mask.any():
                    mask_size = mask.sum().item()
                    final_output[mask] += combined_output[offset:offset+mask_size]
                    offset += mask_size
                    
        return final_output.reshape(x.shape)

class HierarchicalMoEMLPBlock(nn.Module):
    """
    分层MoE结构：
    - 第一级Router：将token分发到不同的专家组
    - 第二级Router：在选定的专家组内选择具体的专家
    优化版本：使用批量处理而不是逐token处理，大幅提升性能
    """
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
        num_expert_groups: int = 6,    # 专家组数量
        experts_per_group: int = 16,   # 每组内的专家数量
        k_groups: int = 2,             # 选择的专家组数量
        k_experts: int = 4,            # 每组内选择的专家数量  
        noisy_gating: bool = True,     # 是否使用噪声门控
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.num_expert_groups = num_expert_groups
        self.experts_per_group = experts_per_group
        self.k_groups = k_groups
        self.k_experts = k_experts
        self.noisy_gating = noisy_gating
        
        # 第一级门控：选择专家组
        self.group_gate = nn.Linear(embedding_dim, num_expert_groups)
        
        # 专家组：每组包含多个专家
        self.expert_groups = nn.ModuleList()
        self.expert_gates = nn.ModuleList()  # 分离门控网络
        
        for group_idx in range(num_expert_groups):
            # 专家网络
            experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embedding_dim, mlp_dim),
                    act(),
                    nn.Linear(mlp_dim, embedding_dim)
                ) for _ in range(experts_per_group)
            ])
            self.expert_groups.append(experts)
            
            # 组内门控网络
            gate = nn.Linear(embedding_dim, experts_per_group)
            self.expert_gates.append(gate)
        
        if self.noisy_gating:
            self.noise_epsilon = 1e-2
    
    def _noisy_top_k_gating(self, x, gate, k):
        """带噪音的Top-K门控机制"""
        clean_logits = gate(x)
        
        if self.noisy_gating and self.training:
            noise = torch.randn_like(clean_logits) * self.noise_epsilon
            logits = clean_logits + noise
        else:
            logits = clean_logits
            
        # 计算top-k门控权重
        top_k_logits, top_k_indices = logits.topk(min(k, logits.size(-1)), dim=-1)
        top_k_gates = F.softmax(top_k_logits, dim=-1)
        
        return top_k_gates, top_k_indices
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """优化的批量处理版本，避免逐token循环"""
        original_shape = x.shape
        x_flat = x.reshape(-1, self.embedding_dim)  # [batch_size, embedding_dim]
        batch_size = x_flat.shape[0]
        
        # 第一级路由：选择专家组（批量处理）
        group_gates, group_indices = self._noisy_top_k_gating(
            x_flat, self.group_gate, self.k_groups
        )  # group_gates: [batch_size, k_groups], group_indices: [batch_size, k_groups]
        
        final_output = torch.zeros_like(x_flat)
        
        # 批量处理：按专家组和专家组合进行分组处理
        for group_k_idx in range(self.k_groups):
            for group_idx in range(self.num_expert_groups):
                # 找出选择了当前专家组的所有token
                group_mask = (group_indices[:, group_k_idx] == group_idx)
                if not group_mask.any():
                    continue
                
                # 批量获取选择当前组的token
                masked_tokens = x_flat[group_mask]  # [num_selected_tokens, embedding_dim]
                masked_group_weights = group_gates[:, group_k_idx][group_mask]  # [num_selected_tokens]
                
                # 过滤权重过小的token
                weight_mask = masked_group_weights >= 1e-6
                if not weight_mask.any():
                    continue
                
                filtered_tokens = masked_tokens[weight_mask]
                filtered_group_weights = masked_group_weights[weight_mask]
                
                # 获取当前专家组
                selected_experts = self.expert_groups[group_idx]
                selected_gate = self.expert_gates[group_idx]
                
                # 第二级路由：在组内选择专家（批量处理）
                expert_gates, expert_indices = self._noisy_top_k_gating(
                    filtered_tokens, selected_gate, self.k_experts
                )  # [num_filtered_tokens, k_experts]
                
                # 批量处理每个专家
                group_output = torch.zeros_like(filtered_tokens)
                for expert_k_idx in range(self.k_experts):
                    for expert_idx in range(self.experts_per_group):
                        # 找出选择了当前专家的所有token
                        expert_mask = (expert_indices[:, expert_k_idx] == expert_idx)
                        if not expert_mask.any():
                            continue
                        
                        # 批量处理选择当前专家的token
                        expert_tokens = filtered_tokens[expert_mask]
                        expert_weights = expert_gates[:, expert_k_idx][expert_mask]
                        
                        # 过滤权重过小的
                        expert_weight_mask = expert_weights >= 1e-6
                        if not expert_weight_mask.any():
                            continue
                        
                        final_expert_tokens = expert_tokens[expert_weight_mask]
                        final_expert_weights = expert_weights[expert_weight_mask]
                        
                        # 批量通过专家网络
                        selected_expert = selected_experts[expert_idx]
                        
                        # 使用梯度检查点减少显存占用
                        if self.training:
                            expert_output = checkpoint(selected_expert, final_expert_tokens, use_reentrant=False)
                        else:
                            expert_output = selected_expert(final_expert_tokens)
                            
                        expert_output = expert_output * final_expert_weights.unsqueeze(-1)
                        
                        # 将结果累加回去
                        cumulative_mask = expert_mask.clone()
                        cumulative_mask[expert_mask] = expert_weight_mask
                        group_output[cumulative_mask] += expert_output
                
                # 应用组权重并累加到最终输出
                group_output = group_output * filtered_group_weights.unsqueeze(-1)
                
                # 将结果映射回原始位置
                cumulative_mask = group_mask.clone() 
                cumulative_mask[group_mask] = weight_mask
                final_output[cumulative_mask] += group_output
        
        return final_output.reshape(original_shape)

class Adapter(nn.Module):
    def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
        super().__init__()
        self.skip_connect = skip_connect
        D_hidden_features = int(D_features * mlp_ratio)
        self.act = act_layer()
        self.D_fc1 = nn.Linear(D_features, D_hidden_features)
        self.D_fc2 = nn.Linear(D_hidden_features, D_features)
        
    def forward(self, x):
        # x is (BT, HW+1, D)
        xs = self.D_fc1(x)
        xs = self.act(xs)
        xs = self.D_fc2(xs)
        if self.skip_connect:
            x = x + xs
        else:
            x = xs
        return x

# class LayerNorm2d1(nn.Module):
#     def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(num_channels))
#         self.bias = nn.Parameter(torch.zeros(num_channels))
#         self.eps = eps

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # 确保输入张量的通道数与权重和偏置的维度匹配
#         u = x.mean(1, keepdim=True)
#         s = (x - u).pow(2).mean(1, keepdim=True)
#         x = (x - u) / torch.sqrt(s + self.eps)
#         # 确保权重和偏置的维度正确扩展
#         x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
#         return x
# class Adapter(nn.Module):
#     def __init__(
#         self, 
#         D_features,  # 输入特征维度
#         mlp_ratio=0.5,  # 进一步降低比例以节省显存
#         act_layer=nn.GELU, 
#         skip_connect=True,
#         dropout=0.1,
#         layer_norm=True,
#         num_layers=2
#     ):
#         super().__init__()
#         self.skip_connect = skip_connect
#         self.D_features = D_features
        
#         # 主干MLP层
#         hidden_dim = int(D_features * mlp_ratio)
        
#         self.norm1 = LayerNorm2d1(D_features) if layer_norm else nn.Identity()
#         self.conv1 = nn.Conv2d(D_features, hidden_dim, 1)
#         self.act = act_layer()
#         self.dropout1 = nn.Dropout(dropout)
#         self.conv2 = nn.Conv2d(hidden_dim, D_features, 1)
#         self.norm2 = LayerNorm2d1(D_features) if layer_norm else nn.Identity()
        
#         self._init_weights()
    
#     def _init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.zeros_(m.bias)
    
#     def forward(self, x):
#         # 输入 x 形状: [B, H, W, C]
#         # 转换为通道优先格式
#         x = x.permute(0, 3, 1, 2)
        
#         # 保存输入用于残差连接
#         identity = x
        
#         # 主干前向传播
#         x = self.norm1(x)
#         x = self.conv1(x)
#         x = self.act(x)
#         x = self.dropout1(x)
#         x = self.conv2(x)
#         x = self.norm2(x)
        
#         # 残差连接
#         if self.skip_connect:
#             x = x + identity
            
#         # 转回原始维度顺序 [B, C, H, W] -> [B, H, W, C]
#         x = x.permute(0, 2, 3, 1)
#         return x 
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class LoraMLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = lora.Linear(embedding_dim, mlp_dim)
        self.lin2 = lora.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
