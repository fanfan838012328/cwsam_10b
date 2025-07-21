"""
Test script for Optimized MoE implementation with efficiency improvements.
"""

import torch
import torch.nn as nn
import time
from models.enhanced_moe import (
    EnhancedMoEMLPBlock, 
    OptimizedMoEMLPBlock, 
    ExpertMonitor,
    create_moe_block
)


def test_optimized_moe_basic():
    """Test basic functionality of Optimized MoE MLP Block."""
    print("Testing Optimized MoE MLP Block basic functionality...")
    
    # Test parameters
    batch_size = 4
    seq_len = 64
    embedding_dim = 768
    mlp_dim = 3072
    num_experts = 64
    k = 4
    
    # Create optimized model
    moe_block = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        enable_parallel_experts=True,
        batch_expert_computation=True,
        enable_expert_monitoring=True
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
    
    # Test performance stats
    perf_stats = moe_block.get_performance_stats()
    print(f"✓ Performance stats: {perf_stats}")
    
    print("Optimized MoE basic functionality test passed!\n")


def test_batch_expert_computation():
    """Test batch expert computation optimization."""
    print("Testing batch expert computation...")
    
    embedding_dim = 512
    mlp_dim = 2048
    num_experts = 32
    k = 4
    
    # Create models with and without batch computation
    moe_batch = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        batch_expert_computation=True,
        enable_parallel_experts=False
    )
    
    moe_no_batch = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        batch_expert_computation=False,
        enable_parallel_experts=False
    )
    
    # Test input
    x = torch.randn(8, 32, embedding_dim)
    
    # Test both implementations
    output_batch, _ = moe_batch(x)
    output_no_batch, _ = moe_no_batch(x)
    
    print(f"✓ Batch computation output shape: {output_batch.shape}")
    print(f"✓ No batch computation output shape: {output_no_batch.shape}")
    
    # Both should produce valid outputs
    assert output_batch.shape == x.shape
    assert output_no_batch.shape == x.shape
    
    print("Batch expert computation test passed!\n")


def test_dynamic_expert_selection():
    """Test dynamic expert selection strategy."""
    print("Testing dynamic expert selection...")
    
    embedding_dim = 256
    mlp_dim = 1024
    num_experts = 64
    
    # Create model with dynamic k
    moe_dynamic = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=4,
        dynamic_k=True,
        min_k=2,
        max_k=8,
        enable_expert_monitoring=True
    )
    
    # Test with different input complexities
    simple_input = torch.ones(4, 16, embedding_dim) * 0.1  # Low complexity
    complex_input = torch.randn(4, 16, embedding_dim) * 2.0  # High complexity
    
    # Forward passes
    simple_output, simple_aux = moe_dynamic(simple_input)
    complex_output, complex_aux = moe_dynamic(complex_input)
    
    print(f"✓ Simple input output shape: {simple_output.shape}")
    print(f"✓ Complex input output shape: {complex_output.shape}")
    
    # Check for dynamic k loss
    if 'dynamic_k_loss' in simple_aux:
        print(f"✓ Simple input dynamic k loss: {simple_aux['dynamic_k_loss'].item():.6f}")
    if 'dynamic_k_loss' in complex_aux:
        print(f"✓ Complex input dynamic k loss: {complex_aux['dynamic_k_loss'].item():.6f}")
    
    print("Dynamic expert selection test passed!\n")


def test_expert_monitoring():
    """Test expert usage monitoring and statistics."""
    print("Testing expert monitoring...")
    
    # Create standalone monitor
    monitor = ExpertMonitor(num_experts=32)
    
    # Simulate expert usage
    for _ in range(10):
        expert_indices = torch.randint(0, 32, (4, 4))  # batch_size=4, k=4
        monitor.update_usage(expert_indices)
        
        # Simulate compute times
        for expert_idx in range(32):
            if torch.rand(1).item() > 0.7:  # 30% chance of being used
                compute_time = torch.rand(1).item() * 0.01
                monitor.update_compute_time(expert_idx, compute_time)
    
    # Get statistics
    usage_stats = monitor.get_usage_distribution()
    compute_stats = monitor.get_compute_time_stats()
    
    print(f"✓ Usage statistics: {usage_stats}")
    print(f"✓ Compute time statistics keys: {list(compute_stats.keys())[:5]}...")  # Show first 5
    
    # Test MoE block monitoring
    moe_block = OptimizedMoEMLPBlock(
        embedding_dim=256,
        mlp_dim=1024,
        num_experts=32,
        k=4,
        enable_expert_monitoring=True
    )
    
    x = torch.randn(2, 16, 256)
    for _ in range(5):
        output, aux_losses = moe_block(x)
    
    perf_stats = moe_block.get_performance_stats()
    print(f"✓ MoE performance stats: {perf_stats}")
    
    print("Expert monitoring test passed!\n")


def test_parallel_expert_computation():
    """Test parallel expert computation (if CUDA available)."""
    print("Testing parallel expert computation...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping parallel computation test")
        return
    
    device = torch.device('cuda')
    embedding_dim = 512
    mlp_dim = 2048
    num_experts = 64
    k = 4
    
    # Create models with and without parallel computation
    moe_parallel = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        enable_parallel_experts=True,
        batch_expert_computation=True
    ).to(device)
    
    moe_sequential = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        enable_parallel_experts=False,
        batch_expert_computation=True
    ).to(device)
    
    # Test input
    x = torch.randn(8, 64, embedding_dim, device=device)
    
    # Warm up
    for _ in range(3):
        _ = moe_parallel(x)
        _ = moe_sequential(x)
    
    # Time parallel computation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        output_parallel, _ = moe_parallel(x)
    torch.cuda.synchronize()
    parallel_time = time.time() - start_time
    
    # Time sequential computation
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(10):
        output_sequential, _ = moe_sequential(x)
    torch.cuda.synchronize()
    sequential_time = time.time() - start_time
    
    print(f"✓ Parallel computation time: {parallel_time:.4f}s")
    print(f"✓ Sequential computation time: {sequential_time:.4f}s")
    print(f"✓ Speedup: {sequential_time/parallel_time:.2f}x")
    
    # Both should produce valid outputs
    assert output_parallel.shape == x.shape
    assert output_sequential.shape == x.shape
    
    print("Parallel expert computation test passed!\n")


def test_performance_comparison():
    """Compare performance between Enhanced and Optimized MoE."""
    print("Testing performance comparison...")
    
    embedding_dim = 768
    mlp_dim = 3072
    num_experts = 64
    k = 4
    batch_size = 4
    seq_len = 64
    
    # Create both models
    enhanced_moe = EnhancedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k
    )
    
    optimized_moe = OptimizedMoEMLPBlock(
        embedding_dim=embedding_dim,
        mlp_dim=mlp_dim,
        num_experts=num_experts,
        k=k,
        enable_parallel_experts=False,  # Fair comparison
        batch_expert_computation=True,
        enable_expert_monitoring=True
    )
    
    # Test input
    x = torch.randn(batch_size, seq_len, embedding_dim)
    
    # Warm up
    for _ in range(3):
        _ = enhanced_moe(x)
        _ = optimized_moe(x)
    
    # Time enhanced MoE
    start_time = time.time()
    for _ in range(20):
        enhanced_output, _ = enhanced_moe(x)
    enhanced_time = time.time() - start_time
    
    # Time optimized MoE
    start_time = time.time()
    for _ in range(20):
        optimized_output, _ = optimized_moe(x)
    optimized_time = time.time() - start_time
    
    print(f"✓ Enhanced MoE time: {enhanced_time:.4f}s")
    print(f"✓ Optimized MoE time: {optimized_time:.4f}s")
    print(f"✓ Speedup: {enhanced_time/optimized_time:.2f}x")
    
    # Both should produce valid outputs
    assert enhanced_output.shape == x.shape
    assert optimized_output.shape == x.shape
    
    print("Performance comparison test passed!\n")


def test_factory_function_optimized():
    """Test factory function with optimized option."""
    print("Testing factory function with optimized option...")
    
    # Create optimized MoE
    moe_optimized = create_moe_block(
        embedding_dim=512,
        mlp_dim=2048,
        num_experts=128,
        k=6,
        optimized=True,
        dynamic_k=True,
        enable_expert_monitoring=True
    )
    
    # Create enhanced MoE
    moe_enhanced = create_moe_block(
        embedding_dim=512,
        mlp_dim=2048,
        num_experts=128,
        k=6,
        optimized=False
    )
    
    x = torch.randn(2, 32, 512)
    
    # Test both
    output_opt, aux_opt = moe_optimized(x)
    output_enh, aux_enh = moe_enhanced(x)
    
    assert output_opt.shape == x.shape
    assert output_enh.shape == x.shape
    
    print(f"✓ Optimized MoE type: {type(moe_optimized).__name__}")
    print(f"✓ Enhanced MoE type: {type(moe_enhanced).__name__}")
    
    # Check if optimized version has additional features
    if hasattr(moe_optimized, 'get_performance_stats'):
        perf_stats = moe_optimized.get_performance_stats()
        print(f"✓ Optimized MoE has performance monitoring: {len(perf_stats) > 0}")
    
    print("Factory function test passed!\n")


if __name__ == "__main__":
    print("Running Optimized MoE Tests...\n")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Run tests
    test_optimized_moe_basic()
    test_batch_expert_computation()
    test_dynamic_expert_selection()
    test_expert_monitoring()
    test_parallel_expert_computation()
    test_performance_comparison()
    test_factory_function_optimized()
    
    print("All optimized MoE tests passed! ✅")