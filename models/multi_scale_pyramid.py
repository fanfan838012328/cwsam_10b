"""
多尺度特征金字塔模块

该模块实现了MultiScaleFeaturePyramid，支持1x、2x、4x多尺度特征提取，
特征融合和上采样机制，专门集成到3B以上的CWSAM模型中。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import math


class AdaptivePoolingPyramid(nn.Module):
    """
    自适应池化金字塔，用于生成多尺度特征
    """
    
    def __init__(self, embed_dim: int, pool_sizes: List[int] = [1, 2, 4]):
        """
        初始化自适应池化金字塔
        
        Args:
            embed_dim: 嵌入维度
            pool_sizes: 池化尺寸列表
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pool_sizes = pool_sizes
        
        # 为每个池化尺寸创建自适应平均池化
        self.adaptive_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(size) for size in pool_sizes
        ])
        
        # 特征投影层
        self.feature_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True)
            ) for _ in pool_sizes
        ])
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, C, H, W]
            
        Returns:
            多尺度特征列表
        """
        features = []
        
        for pool, proj in zip(self.adaptive_pools, self.feature_projs):
            # 自适应池化
            pooled = pool(x)
            # 特征投影
            projected = proj(pooled)
            features.append(projected)
        
        return features


class CrossScaleAttention(nn.Module):
    """
    跨尺度注意力模块，用于融合不同尺度的特征
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8):
        """
        初始化跨尺度注意力
        
        Args:
            embed_dim: 嵌入维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """
        跨尺度注意力融合
        
        Args:
            features: 不同尺度的特征列表 [B, C, H_i, W_i]
            target_size: 目标输出尺寸 (H, W)
            
        Returns:
            融合后的特征 [B, C, H, W]
        """
        B, C, target_H, target_W = features[0].shape[0], self.embed_dim, target_size[0], target_size[1]
        
        # 将所有特征上采样到目标尺寸并转换为序列格式
        feature_tokens = []
        for feat in features:
            # 上采样到目标尺寸
            if feat.shape[-2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            
            # 转换为序列格式 [B, C, H, W] -> [B, H*W, C]
            feat_token = feat.flatten(2).transpose(1, 2)
            feature_tokens.append(feat_token)
        
        # 堆叠所有尺度特征 [B, H*W, num_scales, C]
        all_tokens = torch.stack(feature_tokens, dim=2)
        B, HW, num_scales, C = all_tokens.shape
        
        # 重塑为 [B*H*W, num_scales, C] 以便进行注意力计算
        tokens = all_tokens.view(B * HW, num_scales, C)
        
        # 添加位置编码
        tokens = tokens + self.pos_embed
        
        # 计算注意力
        q = self.q_proj(tokens).view(B * HW, num_scales, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(tokens).view(B * HW, num_scales, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(tokens).view(B * HW, num_scales, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 缩放点积注意力
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B * HW, num_scales, C)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 聚合多尺度特征（加权平均）
        output = output.mean(dim=1)  # [B*H*W, C]
        
        # 重塑回空间格式
        output = output.view(B, target_H, target_W, C).permute(0, 3, 1, 2)
        
        return output


class MultiScaleFeaturePyramid(nn.Module):
    """
    增强的多尺度特征金字塔网络
    
    支持特性：
    - 1x、2x、4x多尺度特征提取
    - 特征融合和上采样机制
    - 跨尺度注意力融合
    - 自适应特征增强
    - 集成到3B以上模型中
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        scales: List[int] = [1, 2, 4],
        fusion_method: str = "attention",  # "attention", "conv", "adaptive"
        use_feature_enhancement: bool = True,
        use_residual_connection: bool = True,
        dropout_rate: float = 0.1,
    ):
        """
        初始化多尺度特征金字塔
        
        Args:
            embed_dim: 嵌入维度
            scales: 尺度列表
            fusion_method: 融合方法 ("attention", "conv", "adaptive")
            use_feature_enhancement: 是否使用特征增强
            use_residual_connection: 是否使用残差连接
            dropout_rate: Dropout率
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.scales = scales
        self.fusion_method = fusion_method
        self.use_feature_enhancement = use_feature_enhancement
        self.use_residual_connection = use_residual_connection
        
        # 多尺度特征提取器
        self.scale_extractors = self._build_scale_extractors()
        
        # 特征融合模块
        self.feature_fusion = self._build_feature_fusion()
        
        # 特征增强模块
        if use_feature_enhancement:
            self.feature_enhancer = self._build_feature_enhancer(dropout_rate)
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim//4),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.Dropout2d(dropout_rate)
        )
        
        # 残差投影（如果输入输出维度不同）
        self.residual_proj = None
        if use_residual_connection:
            self.residual_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        
        self._init_weights()
    
    def _build_scale_extractors(self) -> nn.ModuleDict:
        """构建多尺度特征提取器"""
        extractors = nn.ModuleDict()
        
        for scale in self.scales:
            if scale == 1:
                # 1x尺度使用恒等映射
                extractor = nn.Identity()
            else:
                # 其他尺度使用深度可分离卷积
                extractor = nn.Sequential(
                    # 深度卷积
                    nn.Conv2d(
                        self.embed_dim, self.embed_dim, 
                        kernel_size=3, stride=scale, padding=1,
                        groups=self.embed_dim
                    ),
                    nn.BatchNorm2d(self.embed_dim),
                    nn.GELU(),
                    # 点卷积
                    nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
                    nn.BatchNorm2d(self.embed_dim),
                )
            
            extractors[f'scale_{scale}'] = extractor
        
        return extractors
    
    def _build_feature_fusion(self) -> nn.Module:
        """构建特征融合模块"""
        if self.fusion_method == "attention":
            return CrossScaleAttention(
                embed_dim=self.embed_dim,
                num_heads=max(8, self.embed_dim // 128)
            )
        elif self.fusion_method == "conv":
            return nn.Sequential(
                nn.Conv2d(
                    self.embed_dim * len(self.scales), 
                    self.embed_dim, 
                    kernel_size=1
                ),
                nn.BatchNorm2d(self.embed_dim),
                nn.GELU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.embed_dim),
            )
        elif self.fusion_method == "adaptive":
            return AdaptiveFeatureFusion(self.embed_dim, len(self.scales))
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
    
    def _build_feature_enhancer(self, dropout_rate: float) -> nn.Module:
        """构建特征增强模块"""
        return nn.Sequential(
            # 通道注意力
            ChannelAttention(self.embed_dim),
            # 空间注意力
            SpatialAttention(),
            # 特征细化
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.embed_dim),
            nn.GELU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(self.embed_dim, self.embed_dim, kernel_size=1),
        )
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, H, W, C] (BHWC格式)
            
        Returns:
            增强后的特征 [B, H, W, C]
        """
        # 转换为BCHW格式
        if x.dim() == 4 and x.shape[-1] == self.embed_dim:
            x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        
        B, C, H, W = x.shape
        original_x = x
        
        # 提取多尺度特征
        scale_features = []
        for scale in self.scales:
            extractor = self.scale_extractors[f'scale_{scale}']
            feat = extractor(x)
            scale_features.append(feat)
        
        # 特征融合
        if self.fusion_method == "attention":
            fused_features = self.feature_fusion(scale_features, (H, W))
        elif self.fusion_method == "conv":
            # 上采样所有特征到原始尺寸
            upsampled_features = []
            for feat in scale_features:
                if feat.shape[-2:] != (H, W):
                    feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
                upsampled_features.append(feat)
            
            # 拼接并融合
            concat_features = torch.cat(upsampled_features, dim=1)
            fused_features = self.feature_fusion(concat_features)
        else:
            # adaptive fusion
            fused_features = self.feature_fusion(scale_features, (H, W))
        
        # 特征增强
        if self.use_feature_enhancement:
            enhanced_features = self.feature_enhancer(fused_features)
            fused_features = fused_features + enhanced_features
        
        # 输出投影
        output = self.output_proj(fused_features)
        
        # 残差连接
        if self.use_residual_connection:
            if self.residual_proj is not None:
                residual = self.residual_proj(original_x)
            else:
                residual = original_x
            output = output + residual
        
        # 转换回BHWC格式
        output = output.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        return output
    
    def get_feature_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        获取中间特征图用于可视化和分析
        
        Args:
            x: 输入特征 [B, H, W, C]
            
        Returns:
            包含各尺度特征图的字典
        """
        # 转换为BCHW格式
        if x.dim() == 4 and x.shape[-1] == self.embed_dim:
            x = x.permute(0, 3, 1, 2)
        
        feature_maps = {}
        
        # 提取各尺度特征
        for scale in self.scales:
            extractor = self.scale_extractors[f'scale_{scale}']
            feat = extractor(x)
            feature_maps[f'scale_{scale}'] = feat
        
        return feature_maps


class ChannelAttention(nn.Module):
    """通道注意力模块"""
    
    def __init__(self, embed_dim: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // reduction, embed_dim, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = self.sigmoid(avg_out + max_out)
        return x * attention


class SpatialAttention(nn.Module):
    """空间注意力模块"""
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention_input = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention_input))
        return x * attention


class AdaptiveFeatureFusion(nn.Module):
    """自适应特征融合模块"""
    
    def __init__(self, embed_dim: int, num_scales: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # 学习每个尺度的权重
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
        
        # 特征变换
        self.feature_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
                nn.BatchNorm2d(embed_dim),
                nn.GELU()
            ) for _ in range(num_scales)
        ])
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        )
    
    def forward(self, features: List[torch.Tensor], target_size: Tuple[int, int]) -> torch.Tensor:
        """自适应融合多尺度特征"""
        B, C = features[0].shape[0], self.embed_dim
        H, W = target_size
        
        # 归一化权重
        weights = F.softmax(self.scale_weights, dim=0)
        
        # 加权融合特征
        fused_feature = torch.zeros(B, C, H, W, device=features[0].device)
        
        for i, (feat, weight, transform) in enumerate(zip(features, weights, self.feature_transforms)):
            # 上采样到目标尺寸
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            # 特征变换
            feat = transform(feat)
            
            # 加权累加
            fused_feature += weight * feat
        
        # 最终融合
        output = self.fusion_net(fused_feature)
        
        return output


def create_multi_scale_pyramid(
    embed_dim: int,
    model_size: str,
    **kwargs
) -> Optional[MultiScaleFeaturePyramid]:
    """
    创建多尺度特征金字塔的便捷函数
    
    Args:
        embed_dim: 嵌入维度
        model_size: 模型规模
        **kwargs: 其他参数
        
    Returns:
        MultiScaleFeaturePyramid实例或None（如果不需要）
    """
    # 只有3B以上模型才使用多尺度特征
    if model_size in ["3B", "5B", "7B", "10B"]:
        # 根据模型规模调整配置
        if model_size == "3B":
            fusion_method = "conv"
            scales = [1, 2, 4]
        elif model_size == "5B":
            fusion_method = "attention"
            scales = [1, 2, 4]
        else:  # 7B, 10B
            fusion_method = "attention"
            scales = [1, 2, 4, 8]  # 更多尺度
        
        return MultiScaleFeaturePyramid(
            embed_dim=embed_dim,
            scales=scales,
            fusion_method=fusion_method,
            use_feature_enhancement=True,
            use_residual_connection=True,
            **kwargs
        )
    else:
        return None


if __name__ == "__main__":
    # 测试多尺度特征金字塔
    print("测试MultiScaleFeaturePyramid")
    print("=" * 50)
    
    # 测试不同配置
    configs = [
        {"embed_dim": 1536, "model_size": "3B"},
        {"embed_dim": 2048, "model_size": "5B"},
        {"embed_dim": 3072, "model_size": "10B"},
    ]
    
    for config in configs:
        print(f"\n测试 {config['model_size']} 配置...")
        
        pyramid = create_multi_scale_pyramid(**config)
        if pyramid is not None:
            # 创建测试输入
            B, H, W, C = 2, 64, 64, config['embed_dim']
            x = torch.randn(B, H, W, C)
            
            print(f"输入形状: {x.shape}")
            
            # 前向传播
            with torch.no_grad():
                output = pyramid(x)
                print(f"输出形状: {output.shape}")
                
                # 获取特征图
                feature_maps = pyramid.get_feature_maps(x)
                print(f"特征图数量: {len(feature_maps)}")
                for name, feat in feature_maps.items():
                    print(f"  {name}: {feat.shape}")
        else:
            print("该模型规模不使用多尺度特征金字塔")
    
    print("\n测试完成！")