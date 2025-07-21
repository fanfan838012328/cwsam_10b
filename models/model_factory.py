"""
CWSAM模型工厂和注册系统

该模块提供了根据配置自动创建不同规模CWSAM模型的工厂类，
支持模型规格自动检测和验证，以及模型信息展示功能。
"""

import torch
import torch.nn as nn
from functools import partial
from typing import Dict, List, Optional, Union, Any

from .config import CWSAMScalingConfig, ConfigValidator, ModelScalingConfig
from .models import register
from .sam import SAM_MOE_3B
from .enhanced_moe import EnhancedMoEMLPBlock
from .scalable_image_encoder import ScalableImageEncoderViT
from .scalable_mask_decoder import ScalableMaskDecoder


class ModelFactory:
    """
    CWSAM模型工厂类，根据配置自动创建对应规模的模型
    """
    
    # 模型注册表，将模型规模映射到注册名称
    MODEL_REGISTRY = {
        "1.5B": "sam_moe_1dot5b",
        "2B": "sam_moe_2b",
        "3B": "sam_moe_3b",
        "5B": "sam_moe_5b",
        "7B": "sam_moe_7b",
        "10B": "sam_moe_10b"
    }
    
    def __init__(self):
        """初始化模型工厂"""
        # 注册所有模型规模
        self._register_all_models()
    
    def _register_all_models(self):
        """注册所有预定义规模的模型"""
        # 这里不需要实际实现，因为模型会在create_model时动态注册
        pass
    
    def create_model(self, model_size: str, **kwargs) -> nn.Module:
        """
        根据指定的模型规模创建模型实例
        
        Args:
            model_size: 模型规模 ("1.5B", "2B", "3B", "5B", "7B", "10B")
            **kwargs: 传递给模型的额外参数
            
        Returns:
            创建的模型实例
        """
        # 获取模型配置
        config_manager = CWSAMScalingConfig(model_size)
        config = config_manager.get_config()
        
        # 验证配置
        errors = ConfigValidator.validate_scaling_config(config)
        if errors:
            raise ValueError(f"模型配置验证失败: {errors}")
        
        # 创建模型
        model = self._create_model_from_config(config, **kwargs)
        
        return model
    
    def _create_model_from_config(self, config: ModelScalingConfig, **kwargs) -> nn.Module:
        """
        根据配置创建模型
        
        Args:
            config: 模型配置
            **kwargs: 额外参数
            
        Returns:
            创建的模型实例
        """
        # 提取基本参数
        inp_size = kwargs.get('inp_size', 1024)
        num_classes = kwargs.get('num_classes', 1)
        loss = kwargs.get('loss', 'iou')
        loss_weight = kwargs.get('loss_weight', None)
        ignore_index = kwargs.get('ignore_index', -100)
        
        # 创建编码器配置
        encoder_mode = {
            'name': f'vit_{config.model_size.lower()}',
            'patch_size': config.patch_size,
            'embed_dim': config.embed_dim,
            'depth': config.depth,
            'num_heads': config.num_heads,
            'mlp_ratio': config.mlp_ratio,
            'out_chans': 256,  # 固定输出通道数
            'qkv_bias': True,
            'use_rel_pos': True,
            'window_size': 14,
            'global_attn_indexes': [2, 5, 8, 11] + list(range(12, config.depth)),
            'prompt_embed_dim': 256  # 固定提示嵌入维度
        }
        
        # 创建ScalableSAM模型
        model = ScalableSAM(
            inp_size=inp_size,
            encoder_mode=encoder_mode,
            loss=loss,
            num_classes=num_classes,
            loss_weight=loss_weight,
            ignore_index=ignore_index,
            config=config
        )
        
        return model
    
    def get_model_info(self, model_size: str) -> Dict[str, Any]:
        """
        获取指定规模模型的详细信息
        
        Args:
            model_size: 模型规模
            
        Returns:
            包含模型信息的字典
        """
        config_manager = CWSAMScalingConfig(model_size)
        config = config_manager.get_config()
        
        # 估算参数量和内存使用
        params = config.estimate_parameters()
        memory = config.estimate_memory_usage()
        
        # 收集模型信息
        model_info = {
            "model_size": model_size,
            "parameters": params,
            "parameters_billions": params / 1e9,
            "moe_experts": config.moe_num_experts,
            "depth": config.depth,
            "embed_dim": config.embed_dim,
            "attention_heads": config.num_heads,
            "moe_start_layer": config.moe_start_layer,
            "multi_scale_features": config.use_multi_scale,
            "decoder_depth": config.decoder_depth,
            "memory_usage": {
                "inference_gb": memory["total_inference"],
                "training_gb": memory["total_training"]
            },
            "registration_name": self.MODEL_REGISTRY.get(model_size, f"sam_moe_{model_size.lower()}")
        }
        
        return model_info
    
    def print_model_info(self, model_size: str):
        """
        打印指定规模模型的详细信息
        
        Args:
            model_size: 模型规模
        """
        info = self.get_model_info(model_size)
        
        print(f"\n=== CWSAM {model_size} 模型信息 ===")
        print(f"注册名称: {info['registration_name']}")
        print(f"参数量: {info['parameters']:,} ({info['parameters_billions']:.2f}B)")
        print(f"MoE专家数量: {info['moe_experts']}")
        print(f"模型深度: {info['depth']}")
        print(f"嵌入维度: {info['embed_dim']}")
        print(f"注意力头数: {info['attention_heads']}")
        print(f"MoE开始层: {info['moe_start_layer']}")
        print(f"多尺度特征: {'启用' if info['multi_scale_features'] else '禁用'}")
        print(f"解码器深度: {info['decoder_depth']}")
        print(f"\n内存使用估算:")
        print(f"  推理内存: {info['memory_usage']['inference_gb']:.2f} GB")
        print(f"  训练内存: {info['memory_usage']['training_gb']:.2f} GB")
        print("=" * 40)
    
    def list_available_models(self) -> List[str]:
        """
        列出所有可用的模型规模
        
        Returns:
            模型规模列表
        """
        return list(self.MODEL_REGISTRY.keys())
    
    def detect_model_size(self, model: nn.Module) -> str:
        """
        自动检测模型规模
        
        Args:
            model: 模型实例
            
        Returns:
            检测到的模型规模
        """
        # 计算模型参数量
        params = sum(p.numel() for p in model.parameters())
        params_billions = params / 1e9
        
        # 检查模型结构特征
        if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'blocks'):
            depth = len(model.image_encoder.blocks)
            if hasattr(model.image_encoder, 'embed_dim'):
                embed_dim = model.image_encoder.embed_dim
            else:
                # 尝试从第一个块获取嵌入维度
                embed_dim = model.image_encoder.blocks[0].attn.embed_dim
            
            # 检测MoE专家数量
            moe_experts = 0
            for block in model.image_encoder.blocks:
                if hasattr(block, 'mlp') and hasattr(block.mlp, 'num_experts'):
                    moe_experts = max(moe_experts, block.mlp.num_experts)
        else:
            # 无法从结构检测，仅基于参数量估计
            depth = 0
            embed_dim = 0
            moe_experts = 0
        
        # 根据参数量和结构特征匹配最接近的模型规模
        size_params = {
            "1.5B": 1.5,
            "2B": 2.0,
            "3B": 3.0,
            "5B": 5.0,
            "7B": 7.0,
            "10B": 10.0
        }
        
        # 如果有足够的结构信息，使用结构特征匹配
        if depth > 0 and embed_dim > 0 and moe_experts > 0:
            for size, config in CWSAMScalingConfig.PREDEFINED_CONFIGS.items():
                if (abs(depth - config["depth"]) <= 4 and
                    abs(embed_dim - config["embed_dim"]) <= 256 and
                    abs(moe_experts - config["moe_num_experts"]) <= 16):
                    return size
        
        # 否则使用参数量匹配
        closest_size = "1.5B"
        min_diff = float('inf')
        
        for size, target_params in size_params.items():
            diff = abs(params_billions - target_params)
            if diff < min_diff:
                min_diff = diff
                closest_size = size
        
        return closest_size


# 创建ScalableSAM类，作为所有规模CWSAM模型的基类
@register('scalable_sam')
class ScalableSAM(nn.Module):
    """
    可扩展的SAM模型，支持不同参数规模
    """
    
    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        num_classes=None,
        loss_weight=None,
        ignore_index=-100,
        config=None
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.config = config
        
        # 使用可扩展图像编码器
        self.image_encoder = ScalableImageEncoderViT(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],
            num_heads=encoder_mode['num_heads'],
            mlp_ratio=encoder_mode['mlp_ratio'],
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
            moe_num_experts=config.moe_num_experts,
            moe_k=4,
            moe_noisy_gating=True,
            moe_start_layer_index=config.moe_start_layer,
            use_multi_scale=config.use_multi_scale
        )
        
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        
        # 使用可扩展掩码解码器
        self.mask_decoder = ScalableMaskDecoder(
            num_multimask_outputs=3,
            transformer_depth=config.decoder_depth,
            embedding_dim=self.prompt_embed_dim,
            mlp_dim=2048,
            num_heads=8,
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
        )

        # 初始化其他组件
        self._init_components(loss, ignore_index, loss_weight)
        
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
    
    def _init_components(self, loss, ignore_index, loss_weight):
        """初始化损失函数和位置编码等组件"""
        from .sam import PositionEmbeddingRandom, BBCEWithLogitLoss, IOU
        
        self.loss_mode = loss
        self.ignore_index = ignore_index

        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction='none')
        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss(reduction='none')
        elif self.loss_mode == 'iou':
            if loss_weight is not None:
                pos_weight = torch.tensor(loss_weight, dtype=torch.float)
                self.criterionBCE = torch.nn.CrossEntropyLoss(
                    pos_weight, ignore_index=self.ignore_index
                )
            else:
                self.criterionBCE = torch.nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index
                )
            self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(self.prompt_embed_dim // 2)
        self.no_mask_embed = nn.Embedding(1, self.prompt_embed_dim)
    
    def set_input(self, input, gt_mask):
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self):
        bs = self.input.shape[0]

        # Embed prompts
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_embed_dim), device=self.input.device
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(self.input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        self.pred_mask = masks
        return masks

    def infer(self, input):
        bs = input.shape[0] if len(input.shape) > 3 else 1

        # Embed prompts
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_embed_dim), device=input.device
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)

        # Predict masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Upscale the masks to the original image resolution
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: int,
        original_size: int,
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (int): The size of the image input to the
            model. Used to remove padding.
          original_size (int): The original size of the image
            before resizing for input to the model.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = masks.squeeze(dim=1)
        masks = torch.nn.functional.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size, :input_size]
        masks = torch.nn.functional.interpolate(
            masks, (original_size, original_size), mode="bilinear", align_corners=False
        )
        return masks

    def backward_G(self):
        """Calculate loss for the generator"""
        loss = self.criterionBCE(
            self.pred_mask, torch.argmax(self.gt_mask, dim=1, keepdim=True).squeeze(1)
        )
        self.loss_G = loss
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer.step()  # udpate G's weights

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


# 为每个预定义的模型规模创建注册装饰器
def _create_model_registrations():
    """为每个模型规模创建注册函数"""
    factory = ModelFactory()
    
    for model_size in factory.list_available_models():
        register_name = factory.MODEL_REGISTRY[model_size]
        
        @register(register_name)
        class ModelWrapper(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                self.model = factory.create_model(model_size, **kwargs)
            
            def forward(self, x):
                return self.model(x)
            
            def __getattr__(self, name):
                if name == 'model':
                    return super().__getattr__(name)
                return getattr(self.model, name)


# 自动注册所有模型规模
_create_model_registrations()


# 便捷函数
def create_model(model_size: str, **kwargs) -> nn.Module:
    """创建指定规模的模型的便捷函数"""
    factory = ModelFactory()
    return factory.create_model(model_size, **kwargs)


def get_model_info(model_size: str) -> Dict[str, Any]:
    """获取模型信息的便捷函数"""
    factory = ModelFactory()
    return factory.get_model_info(model_size)


def print_model_info(model_size: str):
    """打印模型信息的便捷函数"""
    factory = ModelFactory()
    factory.print_model_info(model_size)


def list_available_models() -> List[str]:
    """列出所有可用模型规模的便捷函数"""
    factory = ModelFactory()
    return factory.list_available_models()


def detect_model_size(model: nn.Module) -> str:
    """检测模型规模的便捷函数"""
    factory = ModelFactory()
    return factory.detect_model_size(model)


if __name__ == "__main__":
    # 示例用法
    print("CWSAM模型工厂示例")
    print("=" * 50)
    
    factory = ModelFactory()
    
    # 列出所有可用模型
    print("可用模型规模:", factory.list_available_models())
    
    # 打印各个模型的信息
    for size in ["1.5B", "3B", "5B"]:
        factory.print_model_info(size)