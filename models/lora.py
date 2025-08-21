import torch
import torch.nn as nn


class LoRA(nn.Module):
    """Low-Rank Adaptation for the for Query (Q), Key (Q), Value (V) matrices"""

    def __init__(
        self,
        qkv: nn.Module,
        linear_a_q: nn.Module,
        linear_b_q: nn.Module,
        linear_a_v: nn.Module,
        linear_b_v: nn.Module,
        alpha: int | float = None,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(self.dim)
        # LoRA scaling: alpha / r, fallback alpha=r when not provided
        r_q = getattr(self.linear_a_q, 'out_features', None)
        self.alpha = float(alpha if alpha is not None else (r_q if r_q is not None else 1.0))
        self.r = float(r_q if r_q is not None else 1.0)
        self.scaling = self.alpha / self.r

    # Expose linear-like attributes for compatibility with callers that expect nn.Linear
    @property
    def in_features(self):
        return getattr(self.qkv, 'in_features', None)

    @property
    def out_features(self):
        return getattr(self.qkv, 'out_features', None)

    def forward(self, x) -> torch.Tensor:
        # Compute the original qkv
        qkv = self.qkv(x)  # Shape: (B, N, 3 * org_C)

        # Compute the new q and v components
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        # Apply scaling and dtype/device alignment
        new_q = (new_q * self.scaling).to(dtype=qkv.dtype, device=qkv.device)
        new_v = (new_v * self.scaling).to(dtype=qkv.dtype, device=qkv.device)

        # Add new q and v components to the original qkv tensor (supports 2D or 3D shapes)
        qkv[..., : self.dim] += new_q
        qkv[..., -self.dim :] += new_v

        return qkv
