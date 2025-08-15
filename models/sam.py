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

    MaskDecoder,
    TwoWayTransformer_moe,
    ImageEncoderViT_moe_layer,

)

logger = logging.getLogger(__name__)
from .iou_loss import IOU
from typing import Any, Optional, Tuple



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

@register('sam_moe_3b_multitask')
class SAM_MOE_3B_MultiTask(nn.Module):
    """支持多任务的SAM MOE 3B模型"""
    
    def __init__(
        self,
        inp_size=None,
        encoder_mode=None,
        loss=None,
        task_configs=None,
        ignore_index=-100,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = encoder_mode['embed_dim']
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        
        # 共享的图像编码器
        self.image_encoder = ImageEncoderViT_moe_layer(
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
            moe_num_experts=16,
            moe_k=4,
            moe_noisy_gating=True,
            moe_start_layer_index=28
        )

        # 为每个任务创建独立的mask_decoder
        self.task_configs = task_configs
        self.mask_decoders = nn.ModuleDict()
        
        for task_config in task_configs:
            task_id_str = str(task_config['task_id'])
            self.mask_decoders[task_id_str] = MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer_moe(
                    depth=2,
                    embedding_dim=self.prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=self.prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                num_classes=task_config['num_classes'],
            )

        # 为每个任务设置不同的损失函数
        self.loss_mode = loss
        self.ignore_index = ignore_index
        self.loss_functions = {}
        self.task_configs = task_configs  # 保存配置用于后续设备设置
        
        for task_config in task_configs:
            task_id = task_config['task_id']
            
            if loss == 'iou':
                if task_config.get('loss_weight') is not None:
                    pos_weight = torch.tensor(task_config['loss_weight'], dtype=torch.float)
                    # 先不移动到设备，等模型移动到CUDA后再处理
                    self.loss_functions[task_id] = torch.nn.CrossEntropyLoss(
                        weight=pos_weight, ignore_index=ignore_index
                    )
                else:
                    self.loss_functions[task_id] = torch.nn.CrossEntropyLoss(
                        ignore_index=ignore_index
                    )
            elif loss == 'bce':
                self.loss_functions[task_id] = torch.nn.BCEWithLogitsLoss(reduction='none')
            elif loss == 'bbce':
                self.loss_functions[task_id] = BBCEWithLogitLoss()

        # 位置编码
        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

        # EVP相关的参数冻结
        if 'evp' in encoder_mode['name']:
            for k, p in self.named_parameters():
                if (
                    "prompt" not in k
                    and "mask_decoder" not in k
                    and "prompt_encoder" not in k
                ):
                    p.requires_grad = False

    def cuda(self, device=None):
        """重写cuda方法，确保损失函数也移动到GPU"""
        super().cuda(device)
        # 确保损失函数也移动到CUDA
        if hasattr(self, 'loss_functions'):
            for loss_fn in self.loss_functions.values():
                loss_fn.cuda(device)
        return self
    
    def to(self, device):
        """重写to方法，确保损失函数也移动到指定设备"""
        super().to(device)
        self.device = device
        # 确保损失函数也移动到指定设备
        if hasattr(self, 'loss_functions'):
            for loss_fn in self.loss_functions.values():
                loss_fn.to(device)
        return self

    def set_input(self, input, gt_mask, task_ids):
        """设置输入数据，包括任务ID"""
        self.input = input.to(self.device)
        self.gt_mask = gt_mask.to(self.device)
        self.task_ids = task_ids.to(self.device) if not isinstance(task_ids, torch.Tensor) else task_ids.to(self.device)

    def get_dense_pe(self) -> torch.Tensor:
        """获取密集位置编码"""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def forward(self):
        """前向传播 - 支持混合任务批次"""
        bs = self.input.shape[0]
        
        # 1. 共享特征提取
        self.features = self.image_encoder(self.input)
        
        # 2. 找到最大类别数用于padding
        max_classes = 0
        for config in self.task_configs:
            max_classes = max(max_classes, config['num_classes'])
        
        # 3. 根据任务ID路由到不同的解码头 - 直接构建最终tensor
        pred_masks_list = []
        
        for i in range(bs):
            task_id = int(self.task_ids[i].item()) if hasattr(self.task_ids[i], 'item') else int(self.task_ids[i])
            task_id_str = str(task_id)
            
            # 获取对应任务的解码器
            mask_decoder = self.mask_decoders[task_id_str]
            
            # 单样本推理
            sparse_embeddings = torch.empty((1, 0, self.prompt_embed_dim), device=self.input.device)
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                1, -1, self.image_embedding_size, self.image_embedding_size
            )
            
            low_res_masks, _ = mask_decoder(
                image_embeddings=self.features[i:i+1],
                image_pe=self.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            
            # 后处理
            masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
            
            # 获取当前任务的实际类别数
            actual_classes = masks.shape[1]
            
            # 如果当前任务的类别数少于最大值，进行padding
            if actual_classes < max_classes:
                padding_size = max_classes - actual_classes
                padding = torch.zeros(1, padding_size, masks.shape[2], masks.shape[3], 
                                    device=masks.device, dtype=masks.dtype)
                masks = torch.cat([masks, padding], dim=1)
            
            pred_masks_list.append(masks)
        
        # 将所有预测结果拼接（现在都有相同的类别数） - 避免额外存储
        self.pred_mask = torch.cat(pred_masks_list, dim=0)
        del pred_masks_list  # 显式删除临时列表释放内存

    def infer(self, input, task_id=0):
        """推理时指定单一任务ID"""
        bs = input.shape[0]
        task_id_str = str(task_id)
        
        # 共享特征提取
        features = self.image_encoder(input)
        
        # 获取对应任务的解码器
        mask_decoder = self.mask_decoders[task_id_str]
        
        sparse_embeddings = torch.empty((bs, 0, self.prompt_embed_dim), device=input.device)
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )
        
        # 推理时不需要存储iou_predictions，节省显存
        low_res_masks, _ = mask_decoder(
            image_embeddings=features,
            image_pe=self.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        masks = self.postprocess_masks(low_res_masks, self.inp_size, self.inp_size)
        return masks

    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """后处理mask"""
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

    def backward_G(self):
        """计算多任务损失"""
        total_loss = 0
        bs = self.input.shape[0]
        
        # 按任务ID分组计算损失
        unique_task_ids = torch.unique(self.task_ids)
        
        for task_id_tensor in unique_task_ids:
            task_id = int(task_id_tensor.item()) if hasattr(task_id_tensor, 'item') else int(task_id_tensor)
            
            # 找到当前任务的样本索引
            indices = (self.task_ids == task_id).nonzero(as_tuple=True)[0]
            
            if len(indices) > 0:
                # 获取当前任务的预测和真实标签
                task_pred = self.pred_mask[indices]
                task_gt = self.gt_mask[indices]
                
                # 获取当前任务的实际类别数
                task_config = None
                for config in self.task_configs:
                    if config['task_id'] == task_id:
                        task_config = config
                        break
                
                if task_config is not None:
                    actual_num_classes = task_config['num_classes']
                    # 只使用实际类别数的部分，忽略padding
                    task_pred = task_pred[:, :actual_num_classes]
                    task_gt = task_gt[:, :actual_num_classes]
                
                # 使用对应任务的损失函数
                loss_fn = self.loss_functions[task_id]
                
                if self.loss_mode == 'iou':
                    task_loss = loss_fn(task_pred, torch.argmax(task_gt, dim=1))
                else:
                    task_loss = loss_fn(task_pred, task_gt)
                
                total_loss += task_loss
        
        self.loss_G = total_loss
        self.loss_G.backward()

    def optimize_parameters(self):
        """优化参数"""
        self.forward()
        self.optimizer.zero_grad()
        self.backward_G()
        self.optimizer.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """设置网络的梯度需求"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


@register('sam_moe_3b')
class SAM_MOE_3B(nn.Module):
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
        self.image_encoder = ImageEncoderViT_moe_layer(
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
            moe_num_experts=16,
            moe_k=4,
            moe_noisy_gating=True,
            moe_start_layer_index=28
        )
        self.prompt_embed_dim = encoder_mode['prompt_embed_dim']
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer_moe(
                depth=2,
                embedding_dim=self.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=self.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=num_classes,
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
            self.criterionBCE = BBCEWithLogitLoss(reduction='none')

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

        # elif self.loss_mode == 'iou_ce':
        #     self.criterionBCE =  torch.nn.CrossEntropyLoss()
        #     self.criterionIOU = IOU()

        self.pe_layer = PositionEmbeddingRandom(encoder_mode['prompt_embed_dim'] // 2)
        self.inp_size = inp_size
        self.image_embedding_size = inp_size // encoder_mode['patch_size']
        self.no_mask_embed = nn.Embedding(1, encoder_mode['prompt_embed_dim'])

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
        # bs = 1
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

    def infer(self, input):
        bs = 1

        # Embed prompts
        sparse_embeddings = torch.empty(
            (bs, 0, self.prompt_embed_dim), device=input.device
        )
        dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, self.image_embedding_size, self.image_embedding_size
        )

        self.features = self.image_encoder(input)  # 第一个val 第二张图推理循环 显存+5G

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
        # masks_rgb= onehot_to_mask(masks)
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
        # masks = masks[0]
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

    def get_ignore_mask_loss(self, loss, ignore_index: list = None):
        """Create a mask to ignore certain pixels in the ground truth."""
        # 创建一个掩码
        mask = torch.ones_like(loss, device=loss.device)

        # 对于每个要屏蔽的类别，将掩码设置为0
        for index in ignore_index:
            mask[torch.argmax(self.gt_mask, dim=1) == index] = 0

        # 应用掩码到损失上
        loss = loss * mask
        return loss.sum() / torch.ones_like(loss, device=loss.device).sum()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # mask = self.create_ignore_mask(self.gt_mask, ignore_index=self.ignore_index)

        loss = self.criterionBCE(
            self.pred_mask, torch.argmax(self.gt_mask, dim=1, keepdim=True).squeeze(1)
        )  # (1,4,1024,1024)
        # print(
        #     f'未忽略类别loss:{self.criterionBCE(self.pred_mask, self.gt_mask).mean()}'
        # )
        # guanfang_crt = torch.nn.CrossEntropyLoss(ignore_index=0)
        # guanfang_loss = guanfang_crt(
        #     self.pred_mask, torch.argmax(self.gt_mask, dim=1, keepdim=True).squeeze(1)
        # )
        # print(f'guanfang忽略类别loss:{guanfang_loss}')
        # loss = self.get_ignore_mask_loss(loss, ignore_index=self.ignore_index)
        # print(f'忽略类别loss:{loss}')

        # loss = loss * mask  # 应用掩码
        # loss = loss.sum() / mask.sum()  # 仅计算非忽略像素的损失
        self.loss_G = loss
        # if self.loss_mode == 'iou':
        # self.loss_G += _iou_loss(self.pred_mask, self.gt_mask)

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