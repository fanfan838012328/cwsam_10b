#!/usr/bin/env python3
"""
Calculate the number of trainable parameters for DINOv3 model with LoRA adaptation.
Based on the configuration: XinTong_dinov3_r64.yaml
"""

def calculate_dinov3_lora_params(r=64):
    """
    Calculate trainable parameters for DINOv3 ViT-7B with LoRA adaptation.
    
    Args:
        r (int): LoRA rank parameter
    
    Returns:
        dict: Dictionary containing parameter counts
    """
    # DINOv3 ViT-7B model specifications
    embed_dim = 4096  # Hidden dimension from the config (dinov3_vit7b16)
    num_layers = 40   # Number of transformer blocks in ViT-7B
    
    # LoRA parameters per layer
    # Each transformer block has one attention module with qkv projection
    # LoRA is applied to Q and V projections (not K)
    
    # For each layer, we have:
    # - Q projection: Linear(embed_dim, r) + Linear(r, embed_dim) = embed_dim * r + r * embed_dim = 2 * embed_dim * r
    # - V projection: Linear(embed_dim, r) + Linear(r, embed_dim) = embed_dim * r + r * embed_dim = 2 * embed_dim * r
    # Total per layer: 4 * embed_dim * r
    
    params_per_layer = 4 * embed_dim * r
    total_lora_params = num_layers * params_per_layer
    
    # Additional trainable components from the model
    # 1. Projection layers (from dinov3 hidden dim to prompt embed dim)
    dinov3_hidden_dim = 4096
    mid_dim = 1024
    prompt_embed_dim = 256  # From config
    
    projection_params = (
        dinov3_hidden_dim * mid_dim +  # First conv layer
        mid_dim * prompt_embed_dim     # Second conv layer
    )
    
    # 2. MaskDecoder parameters (these are always trainable)
    # This includes the transformer, iou_head, and classification head
    # Approximate calculation based on typical SAM decoder structure
    transformer_dim = prompt_embed_dim  # 256
    mlp_dim = 2048
    num_heads = 8
    transformer_depth = 2
    num_classes = 33  # From config
    
    # TwoWayTransformer parameters (rough estimate)
    transformer_params = (
        transformer_depth * (
            # Self-attention layers
            3 * transformer_dim * transformer_dim +  # qkv projections
            transformer_dim * transformer_dim +      # output projection
            # Cross-attention layers  
            3 * transformer_dim * transformer_dim +  # qkv projections
            transformer_dim * transformer_dim +      # output projection
            # MLP layers
            transformer_dim * mlp_dim + mlp_dim * transformer_dim +
            # Layer norms (approximate)
            4 * transformer_dim
        )
    )
    
    # IOU head parameters
    iou_head_hidden_dim = 256
    iou_head_depth = 3
    iou_head_params = (
        transformer_dim * iou_head_hidden_dim +
        (iou_head_depth - 2) * iou_head_hidden_dim * iou_head_hidden_dim +
        iou_head_hidden_dim * num_classes
    )
    
    # Final classification head
    class_head_params = transformer_dim * num_classes
    
    # Position embedding parameters
    pe_params = 2 * (prompt_embed_dim // 2)  # PositionEmbeddingRandom
    no_mask_embed_params = prompt_embed_dim  # no_mask_embed
    
    mask_decoder_params = transformer_params + iou_head_params + class_head_params + pe_params + no_mask_embed_params
    
    # Total trainable parameters
    total_params = total_lora_params + projection_params + mask_decoder_params
    
    return {
        'lora_rank': r,
        'embed_dim': embed_dim,
        'num_layers': num_layers,
        'lora_params_per_layer': params_per_layer,
        'total_lora_params': total_lora_params,
        'projection_params': projection_params,
        'mask_decoder_params': mask_decoder_params,
        'total_trainable_params': total_params,
        'total_trainable_params_M': total_params / 1e6  # in millions
    }

if __name__ == "__main__":
    # Calculate for r=64 (from the config)
    result = calculate_dinov3_lora_params(r=64)
    
    print("DINOv3 ViT-7B with LoRA (r=64) Trainable Parameters:")
    print(f"LoRA rank (r): {result['lora_rank']}")
    print(f"Embedding dimension: {result['embed_dim']}")
    print(f"Number of transformer layers: {result['num_layers']}")
    print()
    print("Parameter breakdown:")
    print(f"  LoRA parameters per layer: {result['lora_params_per_layer']:,}")
    print(f"  Total LoRA parameters: {result['total_lora_params']:,}")
    print(f"  Projection layer parameters: {result['projection_params']:,}")
    print(f"  Mask decoder parameters: {result['mask_decoder_params']:,}")
    print()
    print(f"Total trainable parameters: {result['total_trainable_params']:,}")
    print(f"Total trainable parameters: {result['total_trainable_params_M']:.2f}M")
    
    # Compare with different r values
    print("\nComparison with different LoRA ranks:")
    for r_val in [16, 32, 64, 128]:
        res = calculate_dinov3_lora_params(r=r_val)
        print(f"  r={r_val:3d}: {res['total_trainable_params_M']:6.2f}M parameters")