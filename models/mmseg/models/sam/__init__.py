# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .sam import Sam
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .mask_decoder_onlycls import MaskDecoder as MaskDecoderOnlyCls
from .prompt_encoder import PromptEncoder
from .transformer import TwoWayTransformer
from .transformer_moe import TwoWayTransformer as TwoWayTransformer_moe
from .image_encoder_vit_adaptor import ImageEncoderViT as ImageEncoderViT_vitadp
from .image_encoder_vit_tiny_adaptor import TinyViT as ImageEncoderViT_tinyadp
from .image_encoder_vit_tiny_lora_mix_adapter import TinyViT as ImageEncoderViT_tiny_lora_mix_adapter
from .image_encoder_vit_adaptor_block_and_block import ImageEncoderViT as ImageEncoderViT_vitadp_b2b
from .image_encoder_vit_lora import ImageEncoderViT as ImageEncoderViT_vitlora
from .image_encoder_vit_lora_mix_adapter import ImageEncoderViT as ImageEncoderViT_vitlora_mix_adapter
# from .image_encoder_vit_multi_scale import ImageEncoderViT as ImageEncoderViT_vit_multi_scale 无效
from .image_encoder_vit_GAM import ImageEncoderViT as ImageEncoderViT_vit_GAM
from .mask_decoder_moe import MaskDecoder as MaskDecoder_moe
from .image_encoder_moe import ImageEncoderViT as ImageEncoderViT_moe
from .image_encoder_moe_layer import ImageEncoderViT as ImageEncoderViT_moe_layer
from .image_encoder_soft_moe import ImageEncoderViT as ImageEncoderViT_soft_moe
from .common import (
    LayerNorm2d,
    MLPBlock,
    Adapter,
    MoEMLPBlock,
    SoftMoEBlock,
)