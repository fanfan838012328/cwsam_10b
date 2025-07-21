"""
可扩展图像编码器实现

该模块实现了ScalableImageEncoderViT类，支持32到64层的可变深度架构，
1280到3072维度的嵌入扩展，以及渐进式MoE层应用。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Type, List, Dict, Any
from functools import partial

try:
    from .config import CWSAMScalingConfig, ModelScalingConfig
    from .enhanced_moe import EnhancedMoEMLPBlock
    from .multi_scale_pyramid import MultiScaleFeaturePyramid, create_multi_scale_pyramid
    from .mmseg.models.sam.common import LayerNorm2d, MLPBlock, Adapter
    from .mmseg.models.sam.image_encoder_moe_layer import (
        PatchEmbed, PromptGenerator, Block, Attention, 
        to_2tuple, trunc_normal_
    )
except ImportError:
    # For standalone execution
    from config import CWSAMScalingConfig, ModelScalingConfig
    from enhanced_moe import EnhancedMoEMLPBlock
    from multi_scale_pyramid import MultiScaleFeaturePyramid, create_multi_scale_pyramid
    from mmseg.models.sam.common import LayerNorm2d, MLPBlock, Adapter
    from mmseg.models.sam.image_encoder_moe_layer import (
        PatchEmbed, PromptGenerator, Block, Attention, 
        to_2tuple, trunc_normal_
    )


class ScalableBlock(nn.Module):
    """
    可扩展的Transformer块，支持动态MoE配置和不同的注意力头数
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        # MoE相关参数
        use_moe: bool = False,
        num_experts: int = 64,
        moe_k: int = 4,
        noisy_gating: bool = True,
        load_balancing: bool = True,
        expert_dropout: float = 0.1,
        # 适配器参数
        use_adapter: bool = True,
    ) -> None:
        """
        初始化可扩展Transformer块
        
        Args:
            dim: 输入通道数
            num_heads: 注意力头数
            mlp_ratio: MLP隐藏层维度比例
            qkv_bias: 是否在QKV中使用偏置
            norm_layer: 归一化层
            act_layer: 激活函数
            use_rel_pos: 是否使用相对位置编码
            rel_pos_zero_init: 是否零初始化相对位置参数
            window_size: 窗口注意力大小
            input_size: 输入分辨率
            use_moe: 是否使用MoE
            num_experts: MoE专家数量
            moe_k: 选择的专家数量
            noisy_gating: 是否使用噪声门控
            load_balancing: 是否使用负载均衡
            expert_dropout: 专家内部dropout率
            use_adapter: 是否使用适配器
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.use_moe = use_moe
        self.use_adapter = use_adapter
        
        # 归一化层
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        
        # 注意力层
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        
        # MLP层 - 根据是否使用MoE选择不同实现
        if use_moe:
            self.mlp = EnhancedMoEMLPBlock(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                num_experts=num_experts,
                k=moe_k,
                act=act_layer,
                noisy_gating=noisy_gating,
                load_balancing=load_balancing,
                expert_dropout=expert_dropout,
            )
        else:
            self.mlp = MLPBlock(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer,
            )
        
        # 适配器层
        if use_adapter:
            self.MLP_Adapter = Adapter(dim, skip_connect=False)
            self.Space_Adapter = Adapter(dim)
        
        # 存储辅助损失
        self.aux_losses = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 注意力分支
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x)
        
        # 空间适配器
        if self.use_adapter:
            x = self.Space_Adapter(x, shortcut)
        else:
            x = shortcut + x
        
        # MLP分支
        shortcut = x
        x = self.norm2(x)
        
        if self.use_moe:
            mlp_output, aux_losses = self.mlp(x)
            self.aux_losses = aux_losses
            x = mlp_output
        else:
            x = self.mlp(x)
        
        # MLP适配器
        if self.use_adapter:
            x = self.MLP_Adapter(x, shortcut)
        else:
            x = shortcut + x
        
        return x
    
    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """获取辅助损失"""
        return self.aux_losses



class ScalableImageEncoderViT(nn.Module):
    """
    可扩展的图像编码器ViT
    
    支持特性：
    - 32到64层的可变深度架构
    - 1280到3072维度的嵌入扩展  
    - 渐进式MoE层应用（从第8层到第28层开始）
    - 动态注意力头数调整（16到48头）
    - 多尺度特征金字塔集成
    """
    
    def __init__(
        self,
        config: CWSAMScalingConfig,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        # 适配器相关参数
        use_prompt_generator: bool = True,
        prompt_scale_factor: int = 32,
        prompt_type: str = 'highpass',
        freq_nums: float = 0.25,
        use_gradient_checkpointing: bool = True,
    ) -> None:
        """
        初始化可扩展图像编码器
        
        Args:
            config: CWSAM扩展配置管理器
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_chans: 输入通道数
            out_chans: 输出通道数
            qkv_bias: QKV是否使用偏置
            norm_layer: 归一化层类型
            act_layer: 激活函数类型
            use_abs_pos: 是否使用绝对位置编码
            use_rel_pos: 是否使用相对位置编码
            rel_pos_zero_init: 是否零初始化相对位置参数
            window_size: 窗口注意力大小
            global_attn_indexes: 全局注意力层索引
            use_prompt_generator: 是否使用提示生成器
            prompt_scale_factor: 提示缩放因子
            prompt_type: 提示类型
            freq_nums: 频率数量
        """
        super().__init__()
        
        # 获取配置
        self.config = config.get_config()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = self.config.embed_dim
        self.depth = self.config.depth
        self.num_heads = self.config.num_heads
        self.moe_start_layer = self.config.moe_start_layer
        self.moe_num_experts = self.config.moe_num_experts
        self.use_multi_scale = self.config.use_multi_scale
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        # 验证配置合理性
        self._validate_config()
        
        # 补丁嵌入
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=self.embed_dim,
        )
        
        # 位置编码
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, self.embed_dim)
            )
        
        # 构建Transformer块
        self.blocks = nn.ModuleList()
        self._build_transformer_blocks(
            norm_layer, act_layer, use_rel_pos, rel_pos_zero_init,
            window_size, global_attn_indexes, qkv_bias
        )
        
        # 多尺度特征金字塔（3B以上模型）
        self.multi_scale_pyramid = None
        if self.use_multi_scale:
            self.multi_scale_pyramid = MultiScaleFeaturePyramid(
                embed_dim=self.embed_dim,
                scales=[1, 2, 4],
                fusion_method="attention"
            )
        
        # 输出颈部网络
        self.neck = self._build_neck(out_chans)
        
        # 提示生成器（适配器）
        self.prompt_generator = None
        if use_prompt_generator:
            self.prompt_generator = PromptGenerator(
                scale_factor=prompt_scale_factor,
                prompt_type=prompt_type,
                embed_dim=self.embed_dim,
                tuning_stage=1234,
                depth=self.depth,
                input_type='fft',
                freq_nums=freq_nums,
                handcrafted_tune=True,
                embedding_tune=True,
                adaptor='adaptor',
                img_size=img_size,
                patch_size=patch_size
            )
        
        # 输出索引
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))
        
        # 初始化权重
        self._init_weights()
        
        # 存储辅助损失
        self.aux_losses = {}
    
    def _validate_config(self):
        """验证配置合理性"""
        # 检查深度范围
        if not (32 <= self.depth <= 64):
            raise ValueError(f"深度 {self.depth} 超出支持范围 [32, 64]")
        
        # 检查嵌入维度范围
        if not (1280 <= self.embed_dim <= 3072):
            raise ValueError(f"嵌入维度 {self.embed_dim} 超出支持范围 [1280, 3072]")
        
        # 检查注意力头数范围
        if not (16 <= self.num_heads <= 48):
            raise ValueError(f"注意力头数 {self.num_heads} 超出支持范围 [16, 48]")
        
        # 检查MoE开始层范围
        if not (8 <= self.moe_start_layer <= 28):
            raise ValueError(f"MoE开始层 {self.moe_start_layer} 超出支持范围 [8, 28]")
        
        # 检查维度整除性
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"嵌入维度 {self.embed_dim} 必须能被注意力头数 {self.num_heads} 整除")
    
    def _build_transformer_blocks(
        self, 
        norm_layer, 
        act_layer, 
        use_rel_pos, 
        rel_pos_zero_init,
        window_size, 
        global_attn_indexes, 
        qkv_bias
    ):
        """构建Transformer块"""
        input_size = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        
        for i in range(self.depth):
            # 确定是否使用MoE
            use_moe = i >= self.moe_start_layer
            
            # 确定窗口大小
            block_window_size = window_size if i not in global_attn_indexes else 0
            
            block = ScalableBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=block_window_size,
                input_size=input_size,
                use_moe=use_moe,
                num_experts=self.moe_num_experts,
                moe_k=4,  # 固定选择4个专家
                noisy_gating=True,
                load_balancing=True,
                expert_dropout=0.1,
                use_adapter=True,
            )
            
            self.blocks.append(block)
    
    def _build_neck(self, out_chans: int) -> nn.Module:
        """构建输出颈部网络"""
        return nn.Sequential(
            nn.Conv2d(
                self.embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )
    
    def _init_weights(self):
        """初始化权重"""
        # 初始化位置编码
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化其他参数
        self.apply(self._init_weights_helper)
    
    def _init_weights_helper(self, m):
        """权重初始化辅助函数"""
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入图像 [B, C, H, W]
            
        Returns:
            编码后的特征 [B, out_chans, H//patch_size, W//patch_size]
        """
        # 保存原始输入用于提示生成
        inp = x
        
        # 补丁嵌入
        x = self.patch_embed(x)  # [B, H//patch_size, W//patch_size, embed_dim]
        
        # 生成提示（如果启用）
        prompts = None
        if self.prompt_generator is not None:
            # 初始化嵌入特征
            embedding_feature = self.prompt_generator.init_embeddings(x)
            # 初始化手工特征
            handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
            # 生成提示
            prompts = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)
        
        # 添加位置编码
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # 通过Transformer块
        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        outs = []
        total_aux_losses = {}
        
        for i, block in enumerate(self.blocks):
            # 添加提示（如果有）
            if prompts is not None:
                prompt = prompts[i].reshape(B, H, W, -1)
                x = x + prompt
            
            # 前向传播（使用梯度检查点节省内存）
            if self.use_gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
            # 收集辅助损失
            aux_losses = block.get_aux_losses()
            for key, value in aux_losses.items():
                if key not in total_aux_losses:
                    total_aux_losses[key] = []
                total_aux_losses[key].append(value)
            
            # 收集输出
            if i in self.out_indices:
                outs.append(x)
        
        # 聚合辅助损失
        self.aux_losses = {}
        for key, losses in total_aux_losses.items():
            if losses:
                self.aux_losses[key] = torch.stack(losses).mean()
        
        # 应用多尺度特征金字塔
        if self.multi_scale_pyramid is not None:
            x = self.multi_scale_pyramid(x)
        
        # 通过颈部网络
        x = self.neck(x.permute(0, 3, 1, 2))  # BHWC -> BCHW
        
        return x
    
    def get_aux_losses(self) -> Dict[str, torch.Tensor]:
        """获取辅助损失"""
        return self.aux_losses
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # 统计MoE层数
        moe_layers = sum(1 for block in self.blocks if block.use_moe)
        
        return {
            "model_size": self.config.model_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "depth": self.depth,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "moe_layers": moe_layers,
            "moe_num_experts": self.moe_num_experts,
            "moe_start_layer": self.moe_start_layer,
            "use_multi_scale": self.use_multi_scale,
            "img_size": self.img_size,
            "patch_size": self.patch_size,
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"\n=== ScalableImageEncoderViT 模型信息 ===")
        print(f"模型规模: {info['model_size']}")
        print(f"总参数量: {info['total_params']:,} ({info['total_params']/1e9:.2f}B)")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"模型深度: {info['depth']}")
        print(f"嵌入维度: {info['embed_dim']}")
        print(f"注意力头数: {info['num_heads']}")
        print(f"MoE层数: {info['moe_layers']}")
        print(f"MoE专家数: {info['moe_num_experts']}")
        print(f"MoE开始层: {info['moe_start_layer']}")
        print(f"多尺度特征: {'启用' if info['use_multi_scale'] else '禁用'}")
        print(f"图像大小: {info['img_size']}x{info['img_size']}")
        print(f"补丁大小: {info['patch_size']}x{info['patch_size']}")
        print("=" * 45)


def create_scalable_image_encoder(
    model_size: str,
    img_size: int = 1024,
    patch_size: int = 16,
    **kwargs
) -> ScalableImageEncoderViT:
    """
    创建可扩展图像编码器的便捷函数
    
    Args:
        model_size: 模型规模 ("1.5B", "2B", "3B", "5B", "7B", "10B")
        img_size: 输入图像大小
        patch_size: 补丁大小
        **kwargs: 其他参数
        
    Returns:
        ScalableImageEncoderViT实例
    """
    try:
        from .config import CWSAMScalingConfig
    except ImportError:
        from config import CWSAMScalingConfig
    
    config = CWSAMScalingConfig(model_size)
    
    return ScalableImageEncoderViT(
        config=config,
        img_size=img_size,
        patch_size=patch_size,
        **kwargs
    )


if __name__ == "__main__":
    # 测试不同规模的编码器
    print("测试ScalableImageEncoderViT")
    print("=" * 50)
    
    for model_size in ["1.5B", "3B", "5B"]:
        try:
            print(f"\n创建 {model_size} 模型...")
            encoder = create_scalable_image_encoder(model_size)
            encoder.print_model_info()
            
            # 测试前向传播
            x = torch.randn(1, 3, 1024, 1024)
            with torch.no_grad():
                output = encoder(x)
                print(f"输出形状: {output.shape}")
                
                # 获取辅助损失
                aux_losses = encoder.get_aux_losses()
                if aux_losses:
                    print(f"辅助损失: {list(aux_losses.keys())}")
                
        except Exception as e:
            print(f"创建 {model_size} 模型失败: {e}")