from torchvision.models.vision_transformer import VisionTransformer
from typing import Optional
import torch.nn as nn
import torch
def vit_h_14(
    image_size: int = 224,
    num_classes: int = 1000,
    pos_embed_size: Optional[int] = None,
    **kwargs
) -> VisionTransformer:
    """
    Modified ViT-Huge-14 model with custom position embedding size
    """
    model = VisionTransformer(
        image_size=image_size,
        patch_size=14,
        num_layers=32,
        num_heads=16,
        hidden_dim=1280,
        mlp_dim=5120,
        num_classes=num_classes,
        **kwargs
    )
    
    # 如果指定了pos_embed_size，修改位置编码的大小
    if pos_embed_size is not None:
        model.encoder.pos_embedding = nn.Parameter(
            torch.zeros(1, pos_embed_size, model.hidden_dim)
        )
        nn.init.normal_(model.encoder.pos_embedding, std=0.02)
    
    return model