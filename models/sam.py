import logging
from functools import partial
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from models import register

# from transformers import ViTModel
from .mmseg.models.sam import (
    ImageEncoderViT,
    MaskDecoder,
    TwoWayTransformer,
    MaskDecoder_moe,
    TwoWayTransformer_moe,
    ImageEncoderViT_moe_layer,
)

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Dict, Optional, Tuple, Union

# 导入可扩展组件
from .config import CWSAMScalingConfig, ModelScalingConfig
from .scalable_image_encoder import ScalableImageEncoderViT
from .scalable_mask_decoder import ScalableMaskDecoder


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x = x.permute(2, 0, 1)
    return x


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)


class BBCEWithLogitLoss(nn.Module):
    '''
    Balanced BCEWithLogitLoss
    '''

    def __init__(self):
        super(BBCEWithLogitLoss, self).__init__()

    def forward(self, pred, gt):
        eps = 1e-10
        count_pos = torch.sum(gt) + eps
        count_neg = torch.sum(1.0 - gt)
        ratio = count_neg / count_pos
        w_neg = count_pos / (count_pos + count_neg)

        bce1 = nn.BCEWithLogitsLoss(pos_weight=ratio)
        loss = w_neg * bce1(pred, gt)

        return loss


def _iou_loss(pred, target):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target).sum(dim=(2, 3)) - inter
    iou = 1 - (inter / union)

    return iou.mean()


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


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
            config=CWSAMScalingConfig(config.model_size),
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            out_chans=encoder_mode['out_chans'],
            qkv_bias=encoder_mode['qkv_bias'],
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            use_rel_pos=encoder_mode['use_rel_pos'],
            rel_pos_zero_init=True,
            window_size=encoder_mode['window_size'],
            global_attn_indexes=encoder_mode['global_attn_indexes'],
        )
        
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        
        # 使用可扩展掩码解码器
        self.mask_decoder = ScalableMaskDecoder(
            config=CWSAMScalingConfig(config.model_size),
            transformer_dim=self.prompt_embed_dim,
            num_multimask_outputs=3,
            activation=nn.GELU,
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        # 获取图像编码器信息
        encoder_info = self.image_encoder.get_model_info()
        
        # 计算总参数量
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_size": self.config.model_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_info": encoder_info,
            "decoder_depth": self.config.decoder_depth,
            "moe_num_experts": self.config.moe_num_experts,
            "embed_dim": self.config.embed_dim,
            "use_multi_scale": self.config.use_multi_scale,
        }
    
    def print_model_info(self):
        """打印模型信息"""
        info = self.get_model_info()
        print(f"\n=== ScalableSAM {info['model_size']} 模型信息 ===")
        print(f"总参数量: {info['total_params']:,} ({info['total_params']/1e9:.2f}B)")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"嵌入维度: {info['embed_dim']}")
        print(f"MoE专家数: {info['moe_num_experts']}")
        print(f"解码器深度: {info['decoder_depth']}")
        print(f"多尺度特征: {'启用' if info['use_multi_scale'] else '禁用'}")
        print("=" * 45)


@register('sam_moe_3b')
class SAM_MOE_3B(nn.Module):
    """
    SAM_MOE_3B类 - 为了向后兼容而保留
    现在是ScalableSAM的包装器，使用3B配置
    """
    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        num_classes=None,
        loss_weight=None,
        ignore_index=-100,
    ):
        super().__init__()
        
        # 创建3B配置
        config = ModelScalingConfig(
            model_size="3B",
            moe_num_experts=64,
            depth=40,
            embed_dim=1536,
            num_heads=24,
            moe_start_layer=20,
            use_multi_scale=True,
            decoder_depth=4
        )
        
        # 创建ScalableSAM实例
        self.model = ScalableSAM(
            inp_size=inp_size,
            encoder_mode=encoder_mode,
            loss=loss,
            num_classes=num_classes,
            loss_weight=loss_weight,
            ignore_index=ignore_index,
            config=config
        )
        
        # 设置设备
        self.device = self.model.device
        
        # 复制关键属性以保持兼容性
        self.image_encoder = self.model.image_encoder
        self.mask_decoder = self.model.mask_decoder
        self.prompt_embed_dim = self.model.prompt_embed_dim
        self.pe_layer = self.model.pe_layer
        self.no_mask_embed = self.model.no_mask_embed
        self.loss_mode = self.model.loss_mode
        self.ignore_index = self.model.ignore_index
        self.criterionBCE = self.model.criterionBCE
        if hasattr(self.model, 'criterionIOU'):
            self.criterionIOU = self.model.criterionIOU
        self.inp_size = self.model.inp_size
        self.image_embedding_size = self.model.image_embedding_size
    
    def set_input(self, input, gt_mask):
        self.model.set_input(input, gt_mask)
        self.input = self.model.input
        self.gt_mask = self.model.gt_mask
    
    def get_dense_pe(self):
        return self.model.get_dense_pe()
    
    def forward(self):
        masks = self.model.forward()
        self.features = self.model.features
        self.pred_mask = self.model.pred_mask
        return masks
    
    def infer(self, input):
        return self.model.infer(input)
    
    def postprocess_masks(self, masks, input_size, original_size):
        return self.model.postprocess_masks(masks, input_size, original_size)
    
    def backward_G(self):
        self.model.backward_G()
        self.loss_G = self.model.loss_G
    
    def optimize_parameters(self):
        self.model.optimize_parameters()
    
    def set_requires_grad(self, nets, requires_grad=False):
        self.model.set_requires_grad(nets, requires_grad)
    
    def __getattr__(self, name):
        """转发未找到的属性到内部模型"""
        if name == 'model':
            return super().__getattr__(name)
        return getattr(self.model, name)