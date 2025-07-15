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
from torch.cuda.amp.autocast_mode import autocast


# from transformers import ViTModel
from .mmseg.models.sam import (
    MaskDecoder_moe,
    TwoWayTransformer_moe,

    # ImageEncoderViT_moe_layer,

    ImageEncoderViT_hierarchical_moe
)

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0).cpu().numpy()
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x = np.transpose(x, (2, 0, 1))  # 使用numpy的transpose代替permute
    return torch.from_numpy(x)


def init_weights(layer):
    if type(layer) == nn.Conv2d:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0.0)
    elif type(layer) == nn.BatchNorm2d:
        # print(layer)
        nn.init.normal_(layer.weight, mean=1.0, std=0.02)
        if layer.bias is not None:
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
        # 确保coords与gaussian matrix的数据类型一致
        coords = coords.to(self.positional_encoding_gaussian_matrix.dtype)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: int) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size, size
        device: Any = self.positional_encoding_gaussian_matrix.device
        # 使用与gaussian matrix相同的数据类型
        dtype = self.positional_encoding_gaussian_matrix.dtype
        grid = torch.ones((h, w), device=device, dtype=dtype)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

# @register('sam_moe_3b')
# class SAM_MOE_3B(nn.Module):
#     def __init__(
#         self,
#         inp_size=None,
#         encoder_mode=None,
#         loss=None,
#         num_classes=None,
#         loss_weight=None,
#         ignore_index=-100,
#     ):
#         super().__init__()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.embed_dim = encoder_mode['embed_dim']
#         self.image_encoder = ImageEncoderViT_moe_layer(
#             img_size=inp_size,
#             patch_size=encoder_mode['patch_size'],
#             in_chans=3,
#             embed_dim=encoder_mode['embed_dim'],
#             depth=encoder_mode['depth'],
#             num_heads=encoder_mode['num_heads'],
#             mlp_ratio=encoder_mode['mlp_ratio'],
#             out_chans=encoder_mode['out_chans'],
#             qkv_bias=encoder_mode['qkv_bias'],
#             norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
#             act_layer=nn.GELU,
#             use_rel_pos=encoder_mode['use_rel_pos'],
#             rel_pos_zero_init=True,
#             window_size=encoder_mode['window_size'],
#             global_attn_indexes=encoder_mode['global_attn_indexes'],
#             moe_num_experts=16,
#             moe_k=4,
#             moe_noisy_gating=True,
#             moe_start_layer_index=28
#         )
#         self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
#         self.mask_decoder = MaskDecoder(
#             num_multimask_outputs=3,
#             transformer=TwoWayTransformer_moe(
#                 depth=2,
#                 embedding_dim=self.prompt_embed_dim,
#                 mlp_dim=2048,
#                 num_heads=8,
#             ),
#             transformer_dim=self.prompt_embed_dim,
#             iou_head_depth=3,
#             iou_head_hidden_dim=256,
#             num_classes=num_classes,
#         )

#         if 'evp' in encoder_mode['name']:
#             for k, p in self.encoder.named_parameters():
#                 if (
#                     "prompt" not in k
#                     and "mask_decoder" not in k
#                     and "prompt_encoder" not in k
#                 ):
#                     p.requires_grad = False

#         self.loss_mode = loss
#         self.ignore_index = ignore_index

#         if self.loss_mode == 'bce':
#             self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction='none')

#         elif self.loss_mode == 'bbce':
#             self.criterionBCE = BBCEWithLogitLoss(reduction='none')

#         elif self.loss_mode == 'iou':
#             # self.criterionBCE = torch.nn.BCEWithLogitsLoss()
#             # pos_weight = torch.tensor([1.5, 1, 0.5, 1.9, 0.1], dtype=torch.float)
#             if loss_weight is not None:
#                 pos_weight = torch.tensor(loss_weight, dtype=torch.float)
#                 self.criterionBCE = torch.nn.CrossEntropyLoss(
#                     pos_weight, ignore_index=self.ignore_index
#                 )
#             else:
#                 self.criterionBCE = torch.nn.CrossEntropyLoss(
#                     ignore_index=self.ignore_index
#                 )

#             self.criterionIOU = IOU()

#         # elif self.loss_mode == 'iou_ce':
#         #     self.criterionBCE =  torch.nn.CrossEntropyLoss()
#         #     self.criterionIOU = IOU()

#         self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
#         self.inp_size = inp_size
#         self.image_embedding_size = inp_size // encoder_mode['patch_size']
#         self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

#     def set_input(self, input, gt_mask):
#         self.input = input.to(self.device)
#         self.gt_mask = gt_mask.to(self.device)

#     def get_dense_pe(self) -> torch.Tensor:
#         """
#         Returns the positional encoding used to encode point prompts,
#         applied to a dense set of points the shape of the image encoding.

#         Returns:
#           torch.Tensor: Positional encoding with shape
#             1x(embed_dim)x(embedding_h)x(embedding_w)
#         """
#         return self.pe_layer(self.image_embedding_size).unsqueeze(0)

#     def forward(self):
#         # bs = 1
#         bs = self.input.shape[0]

#         # Embed prompts
#         sparse_embeddings = torch.empty(
#             (bs, 0, self.prompt_embed_dim), device=self.input.device
#         )
#         dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
#             bs, -1, self.image_embedding_size, self.image_embedding_size
#         )

#         self.features = self.image_encoder(self.input)

#         # Predict masks
#         low_res_masks, iou_predictions = self.mask_decoder(
#             image_embeddings=self.features,
#             image_pe=self.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )

#         # Upscale the masks to the original image resolution
#         masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
#         self.pred_mask = masks

#     def infer(self, input):
#         bs = 1

#         # Embed prompts
#         sparse_embeddings = torch.empty(
#             (bs, 0, self.prompt_embed_dim), device=input.device
#         )
#         dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
#             bs, -1, self.image_embedding_size, self.image_embedding_size
#         )

#         self.features = self.image_encoder(input)  # 第一个val 第二张图推理循环 显存+5G

#         # Predict masks
#         low_res_masks, iou_predictions = self.mask_decoder(
#             image_embeddings=self.features,
#             image_pe=self.get_dense_pe(),
#             sparse_prompt_embeddings=sparse_embeddings,
#             dense_prompt_embeddings=dense_embeddings,
#             multimask_output=False,
#         )

#         # Upscale the masks to the original image resolution
#         masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
#         # masks_rgb= onehot_to_mask(masks)
#         return masks

#     def postprocess_masks(
#         self,
#         masks: torch.Tensor,
#         input_size: Tuple[int, ...],
#         original_size: Tuple[int, ...],
#     ) -> torch.Tensor:
#         """
#         Remove padding and upscale masks to the original image size.

#         Arguments:
#           masks (torch.Tensor): Batched masks from the mask_decoder,
#             in BxCxHxW format.
#           input_size (tuple(int, int)): The size of the image input to the
#             model, in (H, W) format. Used to remove padding.
#           original_size (tuple(int, int)): The original size of the image
#             before resizing for input to the model, in (H, W) format.

#         Returns:
#           (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
#             is given by original_size.
#         """
#         # masks = masks[0]
#         masks = masks.squeeze(dim=1)
#         masks = F.interpolate(
#             masks,
#             (self.image_encoder.img_size, self.image_encoder.img_size),
#             mode="bilinear",
#             align_corners=False,
#         )
#         masks = masks[..., :input_size, :input_size]
#         masks = F.interpolate(
#             masks, original_size, mode="bilinear", align_corners=False
#         )
#         return masks

#     def get_ignore_mask_loss(self, loss, ignore_index: list = None):
#         """Create a mask to ignore certain pixels in the ground truth."""
#         # 创建一个掩码
#         mask = torch.ones_like(loss, device=loss.device)

#         # 对于每个要屏蔽的类别，将掩码设置为0
#         for index in ignore_index:
#             mask[torch.argmax(self.gt_mask, dim=1) == index] = 0

#         # 应用掩码到损失上
#         loss = loss * mask
#         return loss.sum() / torch.ones_like(loss, device=loss.device).sum()

@register('sam_hierarchical_moe_10b')
class SAM_HIERARCHICAL_MOE_10B(nn.Module):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        
        # 使用分层MoE图像编码器
        self.image_encoder = ImageEncoderViT_hierarchical_moe(
            img_size=inp_size,
            patch_size=encoder_mode['patch_size'],
            in_chans=3,
            embed_dim=encoder_mode['embed_dim'],
            depth=encoder_mode['depth'],  # 40层 (垂直扩展)
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
            # 分层MoE参数 (水平扩展: 6组 × 16专家 = 96专家)
            moe_num_expert_groups=encoder_mode.get('moe_num_expert_groups', 6),
            moe_experts_per_group=encoder_mode.get('moe_experts_per_group', 16),
            moe_k_groups=encoder_mode.get('moe_k_groups', 2),
            moe_k_experts=encoder_mode.get('moe_k_experts', 4),
            moe_noisy_gating=encoder_mode.get('moe_noisy_gating', True),
            moe_start_layer_index=encoder_mode.get('moe_start_layer_index', 24),
            use_checkpoint=encoder_mode.get('use_checkpoint', False),
        )
        
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        
        # 定义解码器和Transformer的MoE参数
        decoder_moe_num_experts = encoder_mode.get('decoder_moe_num_experts', 126)
        decoder_moe_k = encoder_mode.get('decoder_moe_k', 2)
        transformer_moe_num_experts = encoder_mode.get('transformer_moe_num_experts', 126)
        transformer_moe_k = encoder_mode.get('transformer_moe_k', 2)

        # 使用MoE的Mask Decoder
        self.mask_decoder = MaskDecoder_moe(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer_moe(
                depth=4,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
                moe_num_experts=transformer_moe_num_experts,
                moe_k=transformer_moe_k,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
            moe_num_experts=decoder_moe_num_experts,
            moe_k=decoder_moe_k,
        )

        if 'evp' in encoder_mode['name']:
            for k, p in self.encoder.named_parameters():
                if (
                    "prompt" not in k
                    and "mask_decoder" not in k
                    and "prompt_encoder" not in k
                ):
                    p.requires_grad = False

        self.loss_mode = loss
        self.ignore_index = ignore_index

        if self.loss_mode == 'bce':
            self.criterionBCE = torch.nn.BCEWithLogitsLoss(reduction='none')

        elif self.loss_mode == 'bbce':
            self.criterionBCE = BBCEWithLogitLoss()

        elif self.loss_mode == 'iou':
            # self.criterionBCE = torch.nn.BCEWithLogitsLoss()
            # pos_weight = torch.tensor([1.5, 1, 0.5, 1.9, 0.1], dtype=torch.float)
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

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

    def set_input(self, input, gt_mask):
        # 输入数据使用autocast自动管理精度，不强制转换
        self.input = input.to(self.device)
        # gt_mask保持原始精度用于损失计算
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

        # 使用autocast进行混合精度前向传播
        with torch.cuda.amp.autocast():
            # Embed prompts - 确保embedding的精度一致
            target_dtype = self.no_mask_embed.weight.dtype
            sparse_embeddings = torch.empty(
                (bs, 0, self.prompt_embed_dim), device=self.input.device, dtype=target_dtype
            )
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size, self.image_embedding_size
            )

            # 使用autocast自动管理精度
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

    def infer(self, input):
        bs = input.shape[0]

        # 使用autocast进行混合精度推理
        with torch.cuda.amp.autocast():
            # Embed prompts - 确保embedding的精度一致
            target_dtype = self.no_mask_embed.weight.dtype
            sparse_embeddings = torch.empty(
                (bs, 0, self.prompt_embed_dim), device=input.device, dtype=target_dtype
            )
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size, self.image_embedding_size
            )

            # 使用autocast自动管理精度
            features = self.image_encoder(input)

            # Predict masks
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=features,
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
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = masks.squeeze(dim=1)
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size, :input_size]
        masks = F.interpolate(
            masks, original_size, mode="bilinear", align_corners=False
        )
        return masks

    def get_ignore_mask_loss(self, loss, ignore_index=None):
        """Create a mask to ignore certain pixels in the ground truth."""
        mask = torch.ones_like(loss, device=loss.device)

        if ignore_index is not None:
            for index in ignore_index:
                mask[torch.argmax(self.gt_mask, dim=1) == index] = 0

        loss = loss * mask
        return loss.sum() / torch.ones_like(loss, device=loss.device).sum()

    def backward_G(self, do_backward=True):
        """Calculate GAN and L1 loss for the generator"""
        # 确保数据类型一致，避免fp16/fp32混合导致的错误
        pred_mask = self.pred_mask
        gt_target = torch.argmax(self.gt_mask, dim=1, keepdim=True).squeeze(1)
        
        # 损失计算使用fp32精度以提高数值稳定性
        # 强制转换预测结果为fp32，确保彻底转换
        with torch.cuda.amp.autocast(enabled=False):
            pred_mask = pred_mask.float()
            
            # 确保target是long类型
            if gt_target.dtype != torch.long:
                gt_target = gt_target.long()
            
            # 如果criterionBCE有权重参数，也需要转为fp32
            if hasattr(self.criterionBCE, 'weight') and self.criterionBCE.weight is not None:
                if self.criterionBCE.weight.dtype != torch.float32:
                    self.criterionBCE.weight.data = self.criterionBCE.weight.data.float()
            
            loss = self.criterionBCE(pred_mask, gt_target)
            
            # 添加MoE负载均衡损失
            load_balance_loss = 0.0
            for name, module in self.named_modules():
                if hasattr(module, 'load_balance_loss'):
                    load_balance_loss += module.load_balance_loss
            
            self.loss_G = loss + load_balance_loss
        
        if do_backward:
            # 使用scaler进行backward，支持混合精度训练
            if hasattr(self, 'scaler') and self.scaler is not None:
                self.scaler.scale(self.loss_G).backward()
            else:
                self.loss_G.backward()

    def optimize_parameters(self, accumulate_steps=1):
        # 兼容梯度累积的优化流程
        # 1. 前向传播
        self.forward()
        
        # 2. 计算损失 (不立即反向传播)
        self.backward_G(do_backward=False)
        
        # 3. 缩放损失
        scaled_loss = self.loss_G / accumulate_steps

        # 4. 反向传播
        if hasattr(self, 'scaler') and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

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