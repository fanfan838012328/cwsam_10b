"""
Test script for Enhanced MoE implementation.
"""

import torch
import torch.nn as nn
from models.enhanced_moe import EnhancedMoEMLPBlock, MoELayer, create_moe_block


def test_enhanced_moe_basic():
    """Test basic functionality of Enhanced MoE MLP Block."""
    print("Testing Enhanced MoE MLP Block basic functionality...")
    
    # Test parameters
    batch_size = 4
    seq_len = 64
    embedding_dim = 768
    mlp_dim = 3072
    num_experts = 32
    k = 4
    
    # Create model
    moe_block = EnhancedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        noisy_gating=True,
        load_balancing=True,
        expert_dropout=0.1
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Forward pass
    output, aux_losses = moe_block(x)
    
    # Check output shape
    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    
    # Check aux losses
    assert 'load_balance_loss' in aux_losses, "Load balance loss not found"
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Load balance loss: {aux_losses['load_balance_loss'].item():.6f}")
    
    # Test expert usage stats
    stats = moe_block.get_expert_usage_stats()
    print(f"✓ Expert usage stats: {stats}")
    
    print("Basic functionality test passed!\n")


def test_different_expert_counts():
    """Test with different expert counts (16 to 512)."""
    print("Testing different expert counts...")
    
    embedding_dim = 512
    mlp_dim = 2048
    batch_size = 2
    seq_len = 32
    
    expert_counts = [16, 32, 64, 128, 256, 512]
    
    for num_experts in expert_counts:
        k = min(4, num_experts)  # Ensure k <= num_experts
        
        moe_block = EnhancedMoEMLPBlock(
            embedding_dim=embedding_dim,
            mlp_dim=mlp_dim,
            num_experts=num_experts,
            k=k
        )
        
        x = torch.randn(batch_size, seq_len, embedding_dim)
        output, aux_losses = moe_block(x)
        
        assert output.shape == x.shape
        print(f"✓ {num_experts} experts: input {x.shape} -> output {output.shape}")
    
    print("Different expert counts test passed!\n")


def test_moe_layer():
    """Test complete MoE layer with normalization."""
    print("Testing MoE Layer...")
    
    embedding_dim = 768
    mlp_dim = 3072
    num_experts = 64
    k = 4
    
    moe_layer = MoELayer(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k
    )
    
    x = torch.randn(2, 64, embedding_dim)
    output, aux_losses = moe_layer(x)
    
    assert output.shape == x.shape
    print(f"✓ MoE Layer: input {x.shape} -> output {output.shape}")
    print("MoE Layer test passed!\n")


def test_factory_function():
    """Test factory function."""
    print("Testing factory function...")
    
    moe_block = create_moe_block(
        embedding_dim=512,
        mlp_dim=2048,
        num_experts=128,
        k=6,
        noisy_gating=True,
        load_balancing=True
    )
    
    x = torch.randn(1, 32, 512)
    output, aux_losses = moe_block(x)
    
    assert output.shape == x.shape
    print(f"✓ Factory function: input {x.shape} -> output {output.shape}")
    print("Factory function test passed!\n")


def test_parameter_validation():
    """Test parameter validation."""
    print("Testing parameter validation...")
    
    # Test invalid expert count
    try:
        EnhancedMoEMLPBlock(
            embedding_dim=512,
            mlp_dim=2048,
            num_experts=8  # Too few
        )
        assert False, "Should have raised ValueError for too few experts"
    except ValueError:
        print("✓ Correctly rejected too few experts")
    
    try:
        EnhancedMoEMLPBlock(
            embedding_dim=512,
            mlp_dim=2048,
            num_experts=1024  # Too many
        )
        assert False, "Should have raised ValueError for too many experts"
    except ValueError:
        print("✓ Correctly rejected too many experts")
    
    # Test k > num_experts
    try:
        EnhancedMoEMLPBlock(
            embedding_dim=512,
            mlp_dim=2048,
            num_experts=32,
            k=64  # k > num_experts
        )
        assert False, "Should have raised ValueError for k > num_experts"
    except ValueError:
        print("✓ Correctly rejected k > num_experts")
    
    print("Parameter validation test passed!\n")


def test_gradient_flow():
    """Test gradient flow through the model."""
    print("Testing gradient flow...")
    
    moe_block = EnhancedMoEMLPBlock(
        embedding_dim=256,
        mlp_dim=1024,
        num_experts=32,
        k=4
    )
    
    x = torch.randn(2, 16, 256, requires_grad=True)
    output, aux_losses = moe_block(x)
    
    # Compute loss
    loss = output.sum() + aux_losses.get('load_balance_loss', 0)
    loss.backward()
    
    # Check gradients
    assert x.grad is not None, "Input gradients not computed"
    
    # Check expert gradients
    expert_grads = []
    for expert in moe_block.experts:
        for param in expert.parameters():
            if param.grad is not None:
                expert_grads.append(param.grad.norm().item())
    
    assert len(expert_grads) > 0, "No expert gradients found"
    print(f"✓ Expert gradient norms: min={min(expert_grads):.6f}, max={max(expert_grads):.6f}")
    
    print("Gradient flow test passed!\n")


if __name__ == "__main__":
    print("Running Enhanced MoE Tests...\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_enhanced_moe_basic()
    test_different_expert_counts()
    test_moe_layer()
    test_factory_function()
    test_parameter_validation()
    test_gradient_flow()
    
    print("All tests passed! ✅")