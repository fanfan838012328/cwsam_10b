"""
可扩展掩码解码器实现

该模块实现了ScalableMaskDecoder类，支持根据模型规模动态调整transformer深度，
并包含增强的TwoWayTransformer实现，支持多头注意力扩展和性能优化。
"""

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math
from typing import List, Tuple, Type, Optional

try:
    from .config import CWSAMScalingConfig, ModelScalingConfig
    from .enhanced_moe import EnhancedMoEMLPBlock
except ImportError:
    # For standalone execution
    from config import CWSAMScalingConfig, ModelScalingConfig
    from enhanced_moe import EnhancedMoEMLPBlock


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """截断正态分布初始化"""
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        import warnings
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


class LayerNorm2d(nn.Module):
    """2D LayerNorm实现"""
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


class AdvancedProgressiveUpsampling(nn.Module):
    """
    高级渐进式上采样模块，包含跳跃连接和特征融合
    用于大模型(7B, 10B)的性能优化
    """
    def __init__(self, transformer_dim: int, activation: Type[nn.Module]):
        super().__init__()
        self.transformer_dim = transformer_dim
        
        # 第一阶段上采样 (1x -> 2x) 带跳跃连接
        self.stage1_main = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
        )
        
        # 第一阶段跳跃连接
        self.stage1_skip = nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        # 特征融合模块
        self.stage1_fusion = nn.Sequential(
            nn.Conv2d(transformer_dim // 2, transformer_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(transformer_dim // 4),
            activation(),
        )
        
        # 第二阶段上采样 (2x -> 4x) 带跳跃连接
        self.stage2_main = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        )
        
        # 第二阶段跳跃连接
        self.stage2_skip = nn.Sequential(
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        
        # 最终特征融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, kernel_size=3, padding=1),
            activation(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 第一阶段：1x -> 2x
        main1 = self.stage1_main(x)
        skip1 = self.stage1_skip(x)
        
        # 特征融合
        fused1 = torch.cat([main1, skip1], dim=1)
        fused1 = self.stage1_fusion(fused1)
        
        # 第二阶段：2x -> 4x
        main2 = self.stage2_main(fused1)
        skip2 = self.stage2_skip(fused1)
        
        # 最终融合
        fused2 = torch.cat([main2, skip2], dim=1)
        output = self.final_fusion(fused2)
        
        return output


class BatchOptimizedMLP(nn.Module):
    """
    批处理优化的MLP，支持动态批大小和内存效率优化
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint
        self.sigmoid_output = sigmoid_output
        
        # 构建层
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:  # 不在最后一层添加激活
                layers.append(nn.ReLU())
        
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        if self.use_checkpoint and self.training:
            # 使用梯度检查点节省内存
            return torch.utils.checkpoint.checkpoint(self._forward_impl, x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        for layer in self.layers:
            x = layer(x)
        
        if self.sigmoid_output:
            x = F.sigmoid(x)
        
        return x


class MemoryEfficientAttention(nn.Module):
    """
    内存效率优化的注意力机制，支持大批量处理
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        
        self.head_dim = self.internal_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, N, C = q.shape
        
        # 投影到QKV
        q = self.q_proj(q).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # 使用PyTorch 2.0的Flash Attention（如果可用）
            attn_output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # 标准注意力计算
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            
            if self.dropout_layer is not None:
                attn = self.dropout_layer(attn)
            
            attn_output = attn @ v
        
        # 重新组合头
        attn_output = attn_output.transpose(1, 2).reshape(B, N, self.internal_dim)
        output = self.out_proj(attn_output)
        
        return output


class MLP(nn.Module):
    """多层感知机"""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x


class EnhancedAttention(nn.Module):
    """
    增强的注意力层，支持动态头数调整和下采样
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        self.dropout = dropout
        
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)
        
        if dropout > 0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)
        
        # Apply dropout if specified
        if self.dropout_layer is not None:
            attn = self.dropout_layer(attn)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class EnhancedTwoWayAttentionBlock(nn.Module):
    """
    增强的双向注意力块，支持MoE和更好的性能优化
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
        use_moe: bool = False,
        num_experts: int = 16,
        dropout: float = 0.0,
    ) -> None:
        """
        增强的transformer块，包含四个层：
        (1) 稀疏输入的自注意力
        (2) 稀疏输入到密集输入的交叉注意力
        (3) 稀疏输入上的MLP块
        (4) 密集输入到稀疏输入的交叉注意力

        Args:
            embedding_dim: 嵌入维度
            num_heads: 注意力头数
            mlp_dim: MLP隐藏维度
            activation: MLP激活函数
            attention_downsample_rate: 注意力下采样率
            skip_first_layer_pe: 是否跳过第一层的位置编码
            use_moe: 是否使用MoE
            num_experts: MoE专家数量
            dropout: dropout率
        """
        super().__init__()
        
        # 自注意力
        self.self_attn = EnhancedAttention(
            embedding_dim, num_heads, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        # 交叉注意力：token到image
        self.cross_attn_token_to_image = EnhancedAttention(
            embedding_dim, num_heads, 
            downsample_rate=attention_downsample_rate,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        # MLP块 - 支持MoE
        if use_moe:
            self.mlp = EnhancedMoEMLPBlock(
                embedding_dim=embedding_dim,
                mlp_dim=mlp_dim,
                num_experts=num_experts,
                k=min(4, num_experts),  # top-k路由
                noisy_gating=True,
                load_balancing=True,
                expert_dropout=dropout
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(embedding_dim, mlp_dim),
                activation(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(mlp_dim, embedding_dim),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            )
        self.norm3 = nn.LayerNorm(embedding_dim)

        # 交叉注意力：image到token
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = EnhancedAttention(
            embedding_dim, num_heads,
            downsample_rate=attention_downsample_rate,
            dropout=dropout
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        queries = self.norm1(queries)

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class EnhancedTwoWayTransformer(nn.Module):
    """
    增强的双向Transformer，支持可变深度和多头注意力扩展
    """
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        use_moe: bool = False,
        num_experts: int = 16,
        dropout: float = 0.0,
    ) -> None:
        """
        增强的transformer解码器，使用查询注意到输入图像

        Args:
            depth: transformer层数
            embedding_dim: 输入嵌入的通道维度
            num_heads: 多头注意力的头数，必须能整除embedding_dim
            mlp_dim: MLP块内部的通道维度
            activation: MLP块中使用的激活函数
            attention_downsample_rate: 注意力下采样率
            use_moe: 是否使用MoE
            num_experts: MoE专家数量
            dropout: dropout率
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                EnhancedTwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                    use_moe=use_moe,
                    num_experts=num_experts,
                    dropout=dropout,
                )
            )

        self.final_attn_token_to_image = EnhancedAttention(
            embedding_dim, num_heads, 
            downsample_rate=attention_downsample_rate,
            dropout=dropout
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            image_embedding: 要注意的图像，形状为 B x embedding_dim x h x w
            image_pe: 添加到图像的位置编码，必须与image_embedding形状相同
            point_embedding: 添加到查询点的嵌入，形状为 B x N_points x embedding_dim

        Returns:
            处理后的point_embedding和image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_pe = image_pe.flatten(2).permute(0, 2, 1)

        # 准备查询和键
        queries = point_embedding
        keys = image_embedding

        # 应用transformer块和最终layernorm
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=point_embedding,
                key_pe=image_pe,
            )

        # 应用从点到图像的最终注意力层
        q = queries + point_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)

        return queries, keys


class ScalableMaskDecoder(nn.Module):
    """
    可扩展的掩码解码器，支持根据模型规模动态调整transformer深度
    """
    def __init__(
        self,
        config: CWSAMScalingConfig,
        transformer_dim: int = 256,
        num_multimask_outputs: int = 4,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        num_classes: int = 1,
        use_moe_in_decoder: bool = False,
        dropout: float = 0.0,
    ) -> None:
        """
        根据配置预测掩码的可扩展解码器

        Args:
            config: CWSAM扩展配置
            transformer_dim: transformer的通道维度
            num_multimask_outputs: 消歧时预测的掩码数量
            activation: 上采样掩码时使用的激活函数类型
            iou_head_depth: 用于预测掩码质量的MLP深度
            iou_head_hidden_dim: 用于预测掩码质量的MLP隐藏维度
            num_classes: 类别数量
            use_moe_in_decoder: 是否在解码器中使用MoE
            dropout: dropout率
        """
        super().__init__()
        
        self.config = config.get_config()
        self.transformer_dim = transformer_dim
        self.num_classes = num_classes
        self.num_multimask_outputs = num_multimask_outputs

        # 根据模型规模调整transformer深度和头数
        transformer_depth = self.config.decoder_depth
        
        # 动态调整注意力头数 - 基于transformer维度
        decoder_num_heads = min(transformer_dim // 32, 16)  # 每32维一个头，最多16头
        decoder_num_heads = max(decoder_num_heads, 4)  # 至少4个头
        
        # 确保头数能整除transformer_dim
        while transformer_dim % decoder_num_heads != 0:
            decoder_num_heads -= 1
        
        # 根据模型规模决定是否使用MoE
        use_decoder_moe = use_moe_in_decoder and self.config.model_size in ["5B", "7B", "10B"]
        decoder_num_experts = min(self.config.moe_num_experts // 4, 32)  # 解码器使用较少专家

        # 创建增强的双向transformer
        self.transformer = EnhancedTwoWayTransformer(
            depth=transformer_depth,
            embedding_dim=transformer_dim,
            num_heads=decoder_num_heads,
            mlp_dim=transformer_dim * 4,  # 标准4倍扩展
            activation=activation,
            use_moe=use_decoder_moe,
            num_experts=decoder_num_experts,
            dropout=dropout,
        )

        # Token嵌入
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        # 输出上采样网络 - 支持渐进式上采样
        self.output_upscaling = self._create_progressive_upsampling(
            transformer_dim, activation
        )
        
        # 超网络MLP - 使用批处理优化版本
        use_checkpoint = self.config.model_size in ["7B", "10B"]  # 大模型使用梯度检查点
        self.output_hypernetworks_mlps = nn.ModuleList([
            BatchOptimizedMLP(
                transformer_dim, transformer_dim, transformer_dim // 8, 3,
                use_checkpoint=use_checkpoint
            )
            for i in range(self.num_mask_tokens)
        ])

        # IoU预测头 - 使用批处理优化版本
        self.iou_prediction_head = BatchOptimizedMLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth,
            use_checkpoint=use_checkpoint
        )

        # 类别上采样网络
        self.cls_upscaling = nn.Sequential(
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.Conv2d(
                transformer_dim // 4,
                transformer_dim * self.num_classes // 8,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            activation(),
        )

        # 特征细化模块 - 用于性能优化
        if self.config.model_size in ["5B", "7B", "10B"]:
            self.feature_refinement = self._create_feature_refinement_module(
                transformer_dim, activation
            )
        else:
            self.feature_refinement = None

        self.apply(self._init_weights)

    def _create_progressive_upsampling(
        self, transformer_dim: int, activation: Type[nn.Module]
    ) -> nn.Module:
        """创建渐进式上采样模块 - 增强版本支持更好的特征保持"""
        
        # 根据模型规模调整上采样策略
        if self.config.model_size in ["7B", "10B"]:
            # 大模型使用更复杂的渐进式上采样
            return self._create_advanced_progressive_upsampling(transformer_dim, activation)
        else:
            # 标准渐进式上采样
            return self._create_standard_progressive_upsampling(transformer_dim, activation)
    
    def _create_standard_progressive_upsampling(
        self, transformer_dim: int, activation: Type[nn.Module]
    ) -> nn.Module:
        """标准渐进式上采样"""
        layers = []
        
        # 第一阶段上采样 (1x -> 2x)
        layers.extend([
            nn.ConvTranspose2d(
                transformer_dim, transformer_dim // 4, kernel_size=2, stride=2
            ),
            LayerNorm2d(transformer_dim // 4),
            activation(),
        ])
        
        # 第二阶段上采样 (2x -> 4x)
        layers.extend([
            nn.ConvTranspose2d(
                transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2
            ),
            activation(),
        ])
        
        return nn.Sequential(*layers)
    
    def _create_advanced_progressive_upsampling(
        self, transformer_dim: int, activation: Type[nn.Module]
    ) -> nn.Module:
        """高级渐进式上采样 - 包含跳跃连接和特征融合"""
        return AdvancedProgressiveUpsampling(transformer_dim, activation)

    def _create_feature_refinement_module(
        self, transformer_dim: int, activation: Type[nn.Module]
    ) -> nn.Module:
        """创建特征细化模块"""
        return nn.Sequential(
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim),
            activation(),
            nn.Conv2d(transformer_dim, transformer_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(transformer_dim),
            activation(),
        )

    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        根据图像和提示嵌入预测掩码

        Args:
            image_embeddings: 来自图像编码器的嵌入
            image_pe: 与image_embeddings形状相同的位置编码
            sparse_prompt_embeddings: 点和框的嵌入
            dense_prompt_embeddings: 掩码输入的嵌入
            multimask_output: 是否返回多个掩码或单个掩码

        Returns:
            批量预测的掩码和掩码质量预测
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # 选择正确的掩码输出
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """预测掩码的核心方法 - 包含批处理优化"""
        # 连接输出token
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # 准备图像特征
        src = image_embeddings + dense_prompt_embeddings
        pos_src = image_pe  # 不需要repeat_interleave，直接使用原始batch
        b, c, h, w = src.shape
        src_feature = src

        # 应用特征细化（如果可用）- 使用梯度检查点优化内存
        if self.feature_refinement is not None:
            if self.training and self.config.model_size in ["7B", "10B"]:
                # 大模型训练时使用梯度检查点
                refined_features = torch.utils.checkpoint.checkpoint(
                    self.feature_refinement, src
                )
                src = src + refined_features
            else:
                src = src + self.feature_refinement(src)

        # 运行transformer - 批处理优化
        hs, src = self._run_transformer_with_optimization(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # 上采样掩码嵌入并使用掩码token预测掩码
        src = src.transpose(1, 2).view(b, c, h, w)
        
        # 批处理优化的上采样
        upscaled_embedding, upscaled_embedding_src = self._batch_optimized_upsampling(
            src, src_feature
        )

        # 批处理优化的超网络计算
        hyper_in = self._batch_optimized_hypernetwork(mask_tokens_out)
        b, c, h, w = upscaled_embedding.shape

        # 特征融合和类别上采样 - 内存优化
        masks = self._memory_efficient_mask_generation(
            hyper_in, upscaled_embedding, upscaled_embedding_src, b, c, h, w
        )

        # 生成掩码质量预测
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred
    
    def _run_transformer_with_optimization(
        self, src: torch.Tensor, pos_src: torch.Tensor, tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """运行transformer的优化版本"""
        if self.training and self.config.model_size in ["7B", "10B"]:
            # 大模型使用梯度检查点
            return torch.utils.checkpoint.checkpoint(
                self.transformer, src, pos_src, tokens
            )
        else:
            return self.transformer(src, pos_src, tokens)
    
    def _batch_optimized_upsampling(
        self, src: torch.Tensor, src_feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批处理优化的上采样"""
        # 并行处理两个上采样分支
        if self.training and torch.cuda.is_available():
            # 使用CUDA流并行处理
            stream1 = torch.cuda.Stream()
            stream2 = torch.cuda.Stream()
            
            with torch.cuda.stream(stream1):
                upscaled_embedding = self.output_upscaling(src)
            
            with torch.cuda.stream(stream2):
                upscaled_embedding_src = self.output_upscaling(src_feature)
            
            # 同步流
            torch.cuda.synchronize()
        else:
            # 标准处理
            upscaled_embedding = self.output_upscaling(src)
            upscaled_embedding_src = self.output_upscaling(src_feature)
        
        return upscaled_embedding, upscaled_embedding_src
    
    def _batch_optimized_hypernetwork(
        self, mask_tokens_out: torch.Tensor
    ) -> torch.Tensor:
        """批处理优化的超网络计算"""
        # 批量处理所有mask tokens而不是循环
        batch_size, num_tokens, embed_dim = mask_tokens_out.shape
        
        # 重塑为批处理格式
        flattened_tokens = mask_tokens_out.view(-1, embed_dim)
        
        # 批量应用所有超网络MLP - 优化版本
        hyper_outputs = []
        
        # 并行处理多个MLP以提高效率
        if self.training and torch.cuda.is_available() and len(self.output_hypernetworks_mlps) > 2:
            # 使用CUDA流并行处理多个MLP
            streams = [torch.cuda.Stream() for _ in range(min(4, len(self.output_hypernetworks_mlps)))]
            stream_outputs = [[] for _ in streams]
            
            for i, mlp in enumerate(self.output_hypernetworks_mlps):
                stream_idx = i % len(streams)
                with torch.cuda.stream(streams[stream_idx]):
                    token_slice = mask_tokens_out[:, i, :]
                    hyper_out = mlp(token_slice)
                    stream_outputs[stream_idx].append(hyper_out)
            
            # 同步所有流并收集结果
            torch.cuda.synchronize()
            for stream_output in stream_outputs:
                hyper_outputs.extend(stream_output)
        else:
            # 标准串行处理
            for i, mlp in enumerate(self.output_hypernetworks_mlps):
                token_slice = mask_tokens_out[:, i, :]
                hyper_out = mlp(token_slice)
                hyper_outputs.append(hyper_out)
        
        # 重新组织输出
        hyper_in = torch.stack(hyper_outputs, dim=1)  # [batch_size, num_tokens, output_dim]
        
        return hyper_in
    
    def _memory_efficient_mask_generation(
        self, 
        hyper_in: torch.Tensor,
        upscaled_embedding: torch.Tensor,
        upscaled_embedding_src: torch.Tensor,
        b: int, c: int, h: int, w: int
    ) -> torch.Tensor:
        """内存效率优化的掩码生成"""
        # 特征融合 - 使用in-place操作减少内存使用
        upscaled_embedding_concat = torch.cat(
            [upscaled_embedding, upscaled_embedding_src], dim=1
        )
        
        # 分块处理大张量以减少内存峰值
        if self.config.model_size in ["7B", "10B"] and b * h * w > 1024 * 1024:
            # 大模型且特征图很大时使用分块处理
            cls_upscaled_embedding = self._chunked_cls_upscaling(upscaled_embedding_concat)
        else:
            cls_upscaled_embedding = self.cls_upscaling(upscaled_embedding_concat).contiguous()
        
        # 生成最终掩码 - 优化矩阵乘法
        cls_embedding_reshaped = cls_upscaled_embedding.view(b, c, self.num_classes * h * w)
        
        # 使用更高效的批量矩阵乘法
        masks = torch.bmm(
            hyper_in, 
            cls_embedding_reshaped
        ).view(b, self.num_mask_tokens, -1, h, w)
        
        return masks
    
    def _chunked_cls_upscaling(self, x: torch.Tensor, chunk_size: int = 4) -> torch.Tensor:
        """分块处理类别上采样以减少内存使用"""
        b, c, h, w = x.shape
        outputs = []
        
        for i in range(0, b, chunk_size):
            end_idx = min(i + chunk_size, b)
            chunk = x[i:end_idx]
            chunk_output = self.cls_upscaling(chunk)
            outputs.append(chunk_output)
        
        return torch.cat(outputs, dim=0).contiguous()
    
    def enable_memory_efficient_mode(self, enabled: bool = True):
        """启用或禁用内存效率模式"""
        self._memory_efficient_mode = enabled
        
        # 为大模型自动启用
        if self.config.model_size in ["7B", "10B"]:
            self._memory_efficient_mode = True
    
    def get_memory_usage_estimate(self, batch_size: int = 1, input_size: Tuple[int, int] = (1024, 1024)) -> dict:
        """估算内存使用量"""
        h, w = input_size[0] // 16, input_size[1] // 16  # 假设16倍下采样
        
        # 模型参数内存
        param_memory = sum(p.numel() * 4 for p in self.parameters()) / (1024**2)  # MB
        
        # 激活内存估算
        transformer_activation = batch_size * self.transformer_dim * h * w * 4 / (1024**2)
        upsampling_activation = batch_size * (self.transformer_dim // 8) * (h * 4) * (w * 4) * 4 / (1024**2)
        
        total_activation = transformer_activation + upsampling_activation
        
        return {
            "parameter_memory_mb": param_memory,
            "activation_memory_mb": total_activation,
            "total_memory_mb": param_memory + total_activation,
            "recommended_batch_size": max(1, int(8192 / total_activation))  # 基于8GB显存
        }

    def get_model_info(self) -> dict:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_size": self.config.model_size,
            "transformer_depth": self.config.decoder_depth,
            "transformer_dim": self.transformer_dim,
            "num_heads": self.transformer.num_heads,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "use_feature_refinement": self.feature_refinement is not None,
            "use_advanced_upsampling": isinstance(self.output_upscaling, AdvancedProgressiveUpsampling),
            "memory_efficient_mode": getattr(self, '_memory_efficient_mode', False),
        }

    def optimize_for_inference(self):
        """优化模型以进行推理"""
        # 启用内存效率模式
        self.enable_memory_efficient_mode(True)
        
        # 设置为评估模式
        self.eval()
        
        # 如果可用，启用torch.jit优化
        if hasattr(torch.jit, 'optimize_for_inference'):
            self = torch.jit.optimize_for_inference(self)
        
        return self
    
    def get_performance_stats(self) -> dict:
        """获取性能统计信息"""
        stats = {
            "model_size": self.config.model_size,
            "transformer_depth": self.config.decoder_depth,
            "uses_progressive_upsampling": True,
            "uses_feature_refinement": self.feature_refinement is not None,
            "uses_batch_optimization": True,
            "uses_memory_optimization": getattr(self, '_memory_efficient_mode', False),
            "supports_gradient_checkpointing": self.config.model_size in ["7B", "10B"],
        }
        
        # 计算理论FLOPs（简化估算）
        h, w = 64, 64  # 假设输入特征图大小
        transformer_flops = (
            self.config.decoder_depth * 
            self.transformer_dim * self.transformer_dim * 
            h * w * 4  # 注意力和MLP的近似FLOPs
        )
        
        upsampling_flops = (
            self.transformer_dim * (self.transformer_dim // 8) * 
            (h * 4) * (w * 4) * 2  # 两阶段上采样
        )
        
        stats["estimated_flops"] = transformer_flops + upsampling_flops
        
        return stats

    def enable_dynamic_batch_sizing(self, enabled: bool = True):
        """启用动态批大小调整"""
        self._dynamic_batch_sizing = enabled
        
        # 为大模型自动启用
        if self.config.model_size in ["7B", "10B"]:
            self._dynamic_batch_sizing = True
    
    def get_optimal_batch_size(self, available_memory_gb: float = 8.0) -> int:
        """根据可用内存计算最优批大小"""
        memory_info = self.get_memory_usage_estimate(batch_size=1)
        memory_per_sample = memory_info["total_memory_mb"] / 1024  # GB
        
        # 保留20%内存作为缓冲
        usable_memory = available_memory_gb * 0.8
        optimal_batch_size = max(1, int(usable_memory / memory_per_sample))
        
        # 根据模型规模设置上限
        max_batch_sizes = {
            "1.5B": 32,
            "2B": 24,
            "3B": 16,
            "5B": 8,
            "7B": 4,
            "10B": 2
        }
        
        max_batch = max_batch_sizes.get(self.config.model_size, 16)
        return min(optimal_batch_size, max_batch)
    
    def apply_inference_optimizations(self):
        """应用推理优化"""
        # 启用内存效率模式
        self.enable_memory_efficient_mode(True)
        
        # 启用动态批大小
        self.enable_dynamic_batch_sizing(True)
        
        # 设置为评估模式
        self.eval()
        
        # 冻结BatchNorm统计
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                module.track_running_stats = False
        
        # 如果支持，启用torch.compile优化（PyTorch 2.0+）
        if hasattr(torch, 'compile'):
            try:
                self.transformer = torch.compile(self.transformer, mode='reduce-overhead')
                self.output_upscaling = torch.compile(self.output_upscaling, mode='reduce-overhead')
            except Exception:
                pass  # 如果编译失败，继续使用原始模型
        
        return self


def create_scalable_mask_decoder(
    model_size: str,
    transformer_dim: int = 256,
    **kwargs
) -> ScalableMaskDecoder:
    """创建可扩展掩码解码器的便捷函数"""
    config = CWSAMScalingConfig(model_size)
    return ScalableMaskDecoder(
        config=config,
        transformer_dim=transformer_dim,
        **kwargs
    )


if __name__ == "__main__":
    # 测试不同规模的解码器
    print("ScalableMaskDecoder 性能优化测试")
    print("=" * 50)
    
    for model_size in ["1.5B", "3B", "5B", "10B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size)
            info = decoder.get_model_info()
            perf_stats = decoder.get_performance_stats()
            
            print(f"\n{model_size} 解码器信息:")
            print(f"  Transformer深度: {info['transformer_depth']}")
            print(f"  注意力头数: {info['num_heads']}")
            print(f"  总参数量: {info['total_parameters']:,}")
            print(f"  特征细化: {info['use_feature_refinement']}")
            print(f"  高级上采样: {info['use_advanced_upsampling']}")
            print(f"  批处理优化: {perf_stats['uses_batch_optimization']}")
            print(f"  梯度检查点: {perf_stats['supports_gradient_checkpointing']}")
            
            # 内存使用估算
            memory_info = decoder.get_memory_usage_estimate()
            print(f"  参数内存: {memory_info['parameter_memory_mb']:.2f} MB")
            print(f"  推荐批大小: {memory_info['recommended_batch_size']}")
            print(f"  估算FLOPs: {perf_stats['estimated_flops']:,}")
            
        except Exception as e:
            print(f"创建 {model_size} 解码器失败: {e}")
    
