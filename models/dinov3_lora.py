import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import LoRA


class DINOV3EncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 32,
        lora_layers: Optional[list[int]] = None,
    ):
        """The DINOv3 encoder-decoder model for finetuning to downstream tasks.
        Args:
            encoder (nn.Module): The ViT encoder model loaded with the DINOv3 model weights.
            r (int, optional): The rank parameter of the LoRA weights. Defaults to 3.
            lora_layers (Optional[list[int]], optional): The list of encoder block layers to apply LoRA. Defaults to None, all layers.
        """
        super().__init__()
        assert r > 0

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Add LoRA layers to the encoder
        if lora_layers is None:
            self.lora_layers = list(range(len(self.encoder.blocks)))
        else:
            self.lora_layers = lora_layers
        
        self.w_a = []
        self.w_b = []

        for i, block in enumerate(self.encoder.blocks):
            if i not in self.lora_layers:
                continue
            w_qkv_linear = block.attn.qkv
            dim = w_qkv_linear.in_features
            
            w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
            w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

            self.w_a.extend([w_a_linear_q, w_a_linear_v])
            self.w_b.extend([w_b_linear_q, w_b_linear_v])

            block.attn.qkv = LoRA(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                alpha=r,
            )
        self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward_features(self, x: torch.Tensor):
        """Proxy to underlying encoder's forward_features for compatibility."""
        return self.encoder.forward_features(x)

    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights to a .pt file
        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}

        w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
        w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

        torch.save({**w_a, **w_b}, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA weights from a file
        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_a):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = nn.Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_b):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = nn.Parameter(saved_tensor)
