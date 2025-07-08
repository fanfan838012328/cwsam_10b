# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# vit adaptor and lowpass fft
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Type

from .common import LayerNorm2d, MLPBlock, Adapter, SoftMoEBlock
import math
import warnings
from itertools import repeat
from soft_moe_pytorch import SoftMoE

import random
import os


TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs

# This class and its supporting functions below lightly adapted from the ViTDet backbone available at: https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/vit.py # noqa
class ImageEncoderViT(nn.Module):
    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
        # MoE相关参数
        moe_num_experts: int = 128,
        moe_k: int = 4,
        moe_noisy_gating: bool = True,
        moe_start_layer_index: int = 0,
        moe_alternate: bool = True,  # 是否在最后几层每隔一层使用MoE
        use_checkpoint: bool = True,  # 新增：是否使用梯度检查点
    ) -> None:
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks.
            global_attn_indexes (list): Indexes for blocks using global attention.
            moe_num_experts (int): MoE专家数量.
            moe_k (int): 每次选择的专家数量.
            moe_noisy_gating (bool): 是否使用噪声门控.
            moe_start_layer_index (int): 从第几层开始应用MoE (0表示所有层).
            moe_alternate (bool): 是否在最后几层每隔一层使用MoE.
            use_checkpoint (bool): Whether to use gradient checkpointing.
        """
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.moe_start_layer_index = moe_start_layer_index
        self.moe_alternate = moe_alternate
        self.use_checkpoint = use_checkpoint

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim)
            )

        self.blocks = nn.ModuleList()
        for i in range(depth): 
            # 确定是否在当前层使用MoE
            # 如果moe_alternate为True，则在满足起始层条件且层索引为奇数时使用MoE
            use_moe_layer = (i >= self.moe_start_layer_index)
            if self.moe_alternate and use_moe_layer:
                use_moe_layer = ((i - self.moe_start_layer_index) % 2 == 0)
                
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
                num_experts=moe_num_experts,
                k=moe_k,
                noisy_gating=moe_noisy_gating,
                use_moe=use_moe_layer
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
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

        #adaptor
        self.scale_factor = 32
        self.prompt_type = 'highpass'
        self.tuning_stage = 1234
        self.input_type = 'fft'
        self.freq_nums = 0.25
        self.handcrafted_tune = True
        self.embedding_tune = True
        self.adaptor = 'adaptor'
        self.prompt_generator = PromptGenerator(self.scale_factor, self.prompt_type, self.embed_dim,
                                                self.tuning_stage, self.depth,
                                                self.input_type, self.freq_nums,
                                                self.handcrafted_tune, self.embedding_tune, self.adaptor,
                                                img_size, patch_size)
        self.num_stages = self.depth
        self.out_indices = tuple(range(self.num_stages))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.patch_embed(x)
        
        # 确保输入需要梯度
        if self.training:
            x.requires_grad_(True)
        
        embedding_feature = self.prompt_generator.init_embeddings(x)
        handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        prompt = self.prompt_generator.get_prompt(handcrafted_feature, embedding_feature)
        
        if self.pos_embed is not None:
            x = x + self.pos_embed

        B, H, W = x.shape[0], x.shape[1], x.shape[2]
        outs = []

        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                def create_custom_forward(blk, prompt_i):
                    def custom_forward(*inputs):
                        x_tensor = inputs[0]
                        return blk(prompt_i.reshape(B, H, W, -1) + x_tensor)
                    return custom_forward

                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(blk, prompt[i]),
                    x,
                    use_reentrant=False,
                    preserve_rng_state=True
                )
            else:
                x = prompt[i].reshape(B, H, W, -1) + x
                x = blk(x)

            if i in self.out_indices:
                outs.append(x)

        x = self.neck(x.permute(0, 3, 1, 2))
        return x
    
    def vis_handcrafted(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x = self.patch_embed(x)
        ################# adaptor
        #embedding_feature = self.prompt_generator.init_embeddings(x)
        #handcrafted_feature = self.prompt_generator.init_handcrafted(inp)
        inv = self.fft(inp, self.freq_nums)
        array = inv.squeeze().cpu().numpy()
        
        array = (array * 255).astype(np.uint8)
        array = np.transpose(array, (1, 2, 0))
        image = Image.fromarray(array, mode='RGB')
        image.save('/remote-home/pxy/CWSAM/vis_fft/vis_after_fft.jpg')
        

def to_2tuple(x):
    if isinstance(x, container_abcs.Iterable):
        return x
    return tuple(repeat(x, 2))

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class PromptGenerator(nn.Module):
    def __init__(self, scale_factor, prompt_type, embed_dim, tuning_stage, depth, input_type,
                 freq_nums, handcrafted_tune, embedding_tune, adaptor, img_size, patch_size):
        """
        Args:
        """
        super(PromptGenerator, self).__init__()
        self.scale_factor = scale_factor
        self.prompt_type = prompt_type
        self.embed_dim = embed_dim
        self.input_type = input_type
        self.freq_nums = freq_nums
        self.tuning_stage = tuning_stage
        self.depth = depth
        self.handcrafted_tune = handcrafted_tune
        self.embedding_tune = embedding_tune
        self.adaptor = adaptor

        self.shared_mlp = nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
        self.embedding_generator = nn.Linear(self.embed_dim, self.embed_dim//self.scale_factor)
        for i in range(self.depth):

            lightweight_mlp = nn.Sequential(
                nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor),
                nn.GELU(),
                #nn.Linear(self.embed_dim//self.scale_factor, self.embed_dim)
            )



            ###### add by pxy 230706 #########
            # lightweight_mlp = MLP(self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor,self.embed_dim//self.scale_factor, self.embed_dim//self.scale_factor,3)

            ###########


            setattr(self, 'lightweight_mlp_{}'.format(str(i)), lightweight_mlp)

        self.prompt_generator = PatchEmbed2(img_size=img_size,
                                                   patch_size=patch_size, in_chans=3,
                                                   embed_dim=self.embed_dim//self.scale_factor)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
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

    def init_embeddings(self, x):
        N, C, H, W = x.permute(0, 3, 1, 2).shape  #[1,64,64,768]
        x = x.reshape(N, C, H*W).permute(0, 2, 1) #[1,4096,768]
        return self.embedding_generator(x)

    def init_handcrafted(self, x):
        x = self.fft(x, self.freq_nums) #[1,3,1024,1024]
        return self.prompt_generator(x)

    def get_prompt(self, handcrafted_feature, embedding_feature):
        N, C, H, W = handcrafted_feature.shape
        handcrafted_feature = handcrafted_feature.view(N, C, H*W).permute(0, 2, 1) #[1,4096,24]
        prompts = []
        for i in range(self.depth):
            lightweight_mlp = getattr(self, 'lightweight_mlp_{}'.format(str(i)))
            # prompt = proj_prompt(prompt)
            prompt = lightweight_mlp(handcrafted_feature + embedding_feature) #[1,4096,24]
            prompts.append(self.shared_mlp(prompt))  
            #prompts.append(prompt)
        return prompts #12*[1,4096,768]

    def forward(self, x):
        if self.input_type == 'laplacian':
            pyr_A = self.lap_pyramid.pyramid_decom(img=x, num=self.freq_nums)
            x = pyr_A[:-1]
            laplacian = x[0]
            for x_i in x[1:]:
                x_i = F.interpolate(x_i, size=(laplacian.size(2), laplacian.size(3)), mode='bilinear', align_corners=True)
                laplacian = torch.cat([laplacian, x_i], dim=1)
            x = laplacian
        elif self.input_type == 'fft':
            x = self.fft(x, self.freq_nums)
        elif self.input_type == 'all':
            x = self.prompt.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)

        # get prompting
        prompt = self.prompt_generator(x)

        if self.mode == 'input':
            prompt = self.proj(prompt)
            return prompt
        elif self.mode == 'stack':
            prompts = []
            for i in range(self.depth):
                proj = getattr(self, 'proj_{}'.format(str(i)))
                prompts.append(proj(prompt))
            return prompts
        elif self.mode == 'hierarchical':
            prompts = []
            for i in range(self.depth):
                proj_prompt = getattr(self, 'proj_prompt_{}'.format(str(i)))
                prompt = proj_prompt(prompt)
                prompts.append(self.proj_token(prompt))
            return prompts


    def fft(self, x_orig: torch.Tensor, rate: float) -> torch.Tensor:
        original_dtype = x_orig.dtype
        
        # 预先检查输入
        if torch.isnan(x_orig).any() or torch.isinf(x_orig).any():
            print(f"警告：输入张量包含NaN或Inf值 - 尝试修复")
            x_tmp = x_orig.clone()
            x_tmp = torch.nan_to_num(x_tmp, nan=0.0, posinf=0.0, neginf=0.0)
            x_orig = x_tmp
            
        # 保证rate是合理值
        rate = max(0.01, min(0.99, rate))
        
        try:
            # 确保 FFT 操作在 float32 精度下进行，以保持数值稳定性
            with torch.cuda.amp.autocast(enabled=False):
                x = x_orig.to(torch.float32)  # 始终使用 float32 进行 FFT 计算
                
                # 预处理：确保输入值在合理范围内
                if x.min() < -1e6 or x.max() > 1e6:
                    print(f"警告：输入值范围异常 min={x.min()}, max={x.max()}, 正在裁剪")
                    x = torch.clamp(x, -1e6, 1e6)
                    
                # 创建掩码
                mask = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                w, h = x.shape[-2:]
                line = max(1, int((w * h * rate) ** .5 // 2))
                mask[:, :, w//2-line:w//2+line, h//2-line:h//2+line] = 1
                
                try:
                    fft = torch.fft.fftshift(torch.fft.fft2(x, norm="forward"))
                except RuntimeError as e:
                    print(f"FFT操作失败：{e}")
                    return torch.zeros_like(x_orig)
                
                if torch.isnan(fft).any() or torch.isinf(fft).any():
                    print(f"FFT结果包含NaN或Inf - 输入统计: min={x.min().item()}, max={x.max().item()}, mean={x.mean().item()}")
                    return torch.zeros_like(x_orig)
                
                # 低通滤波
                fft_low = fft * mask
                fr_low = fft_low.real
                fi_low = fft_low.imag
                
                if torch.isnan(fr_low).any() or torch.isnan(fi_low).any():
                    print("低频实部或虚部包含NaN")
                    return torch.zeros_like(x_orig)
                
                fft_hires_low = torch.fft.ifftshift(torch.complex(fr_low, fi_low))
                
                try:
                    inv_low_complex = torch.fft.ifft2(fft_hires_low, norm="forward")
                    inv_low = inv_low_complex.real
                except RuntimeError as e:
                    print(f"逆FFT操作失败：{e}")
                    return torch.zeros_like(x_orig)
                
                if torch.isnan(inv_low).any() or torch.isinf(inv_low).any():
                    print(f"逆FFT结果包含NaN - fft_hires_low统计: min={fft_hires_low.abs().min().item()}, max={fft_hires_low.abs().max().item()}")
                    return torch.zeros_like(x_orig)
                
                # 取绝对值
                inv_low = torch.abs(inv_low)
                
                if torch.isnan(inv_low).any() or torch.isinf(inv_low).any():
                    print("绝对值操作后出现NaN")
                    return torch.zeros_like(x_orig)
                
                # 返回处理后的结果，转换回原始数据类型
                processed_output = inv_low
                
        except Exception as e:
            print(f"FFT处理中发生异常: {e}")
            return torch.zeros_like(x_orig)
            
        # 最后的安全检查
        result = processed_output.to(original_dtype)
        if torch.isnan(result).any() or torch.isinf(result).any():
            print("返回前检测到NaN，使用零张量替代")
            return torch.zeros_like(x_orig)
            
        return result

class PatchEmbed2(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)#,groups=2)

    def forward(self, x):
        B, C, H, W = x.shape #1,3,1024,1024
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)  #[1,24,64,64]
        return x
########## adaptor end

class Block(nn.Module):
    """Transformer blocks with support of window attention and residual propagation blocks"""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        scale: float = 0.5,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
        num_experts: int = 32,     # MoE专家数量
        k: int = 4,                # 每次选择的专家数量
        noisy_gating: bool = True, # 是否使用噪声门控
        use_moe: bool = True,      # 新增：是否在本层使用MoE
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            window_size (int): Window size for window attention blocks. If it equals 0, then
                use global attention.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
            num_experts (int): 专家数量.
            k (int): 每次选择的专家数量.
            noisy_gating (bool): 是否使用噪声门控.
            use_moe (bool): 是否在本层使用MoE MLP.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.MLP_Adapter = Adapter(dim, skip_connect=False)  # MLP-adapter, no skip connection
        self.Space_Adapter = Adapter(dim)  # with skip connection
        self.scale = scale
        self.Depth_Adapter = Adapter(dim, skip_connect=False)  # no skip connection

        self.norm2 = norm_layer(dim)
        self.use_moe = use_moe
        if self.use_moe:
            # 使用SoftMoEBlock替代MoEMLPBlock
            # 计算序列长度，这里通常是H*W
            seq_len = input_size[0] * input_size[1] if input_size else None
            
            self.mlp = SoftMoEBlock(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer,
                num_experts=num_experts,
                # 使用动态槽位SoftMoE，不需要指定seq_len
                dynamic_slots=True,
                geglu=True
            )
        else:
            self.mlp = MLPBlock(
                embedding_dim=dim,
                mlp_dim=int(dim * mlp_ratio),
                act=act_layer
            )

        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 确保输入需要梯度
        if self.training:
            x.requires_grad_(True)
            
        shortcut = x
        x = self.norm1(x)
        
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        x = self.Space_Adapter(x)
        
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + x
        xn = self.norm2(x)
        x = x + self.mlp(xn) + self.scale * self.MLP_Adapter(xn)
        
        return x


class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool):  If True, add a learnable bias to query, key, value.
            rel_pos (bool): If True, add relative positional embeddings to the attention map.
            rel_pos_zero_init (bool): If True, zero initialize relative positional parameters.
            input_size (tuple(int, int) or None): Input resolution for calculating the relative
                positional parameter size.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert (
                input_size is not None
            ), "Input size must be provided if using relative positional encoding."
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        windows (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = (16, 16),
        stride: Tuple[int, int] = (16, 16),
        padding: Tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x



# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
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
