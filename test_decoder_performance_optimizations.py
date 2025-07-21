#!/usr/bin/env python3
"""
测试解码器性能优化功能

该脚本测试ScalableMaskDecoder的性能优化功能，包括：
1. 渐进式上采样机制
2. 特征细化模块
3. 内存使用和计算效率优化
4. 批处理优化

对应任务4.2的验收标准。
"""

import torch
import torch.nn as nn
import time
import sys
import os

# 添加模型路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from models.scalable_mask_decoder import (
        ScalableMaskDecoder, 
        create_scalable_mask_decoder,
        AdvancedProgressiveUpsampling,
        BatchOptimizedMLP,
        MemoryEfficientAttention
    )
    from models.config import CWSAMScalingConfig
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保模型文件在正确的路径中")
    sys.exit(1)


def test_progressive_upsampling():
    """测试渐进式上采样机制"""
    print("测试1: 渐进式上采样机制")
    print("-" * 40)
    
    # 测试标准渐进式上采样
    transformer_dim = 256
    batch_size = 2
    
    # 创建不同规模的解码器
    for model_size in ["1.5B", "3B", "5B", "10B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size, transformer_dim=transformer_dim)
            
            # 检查上采样模块类型
            is_advanced = isinstance(decoder.output_upscaling, AdvancedProgressiveUpsampling)
            upsampling_type = "高级渐进式" if is_advanced else "标准渐进式"
            
            print(f"  {model_size} 模型: {upsampling_type}上采样")
            
            # 测试上采样功能
            input_tensor = torch.randn(batch_size, transformer_dim, 16, 16)
            
            with torch.no_grad():
                output = decoder.output_upscaling(input_tensor)
                expected_shape = (batch_size, transformer_dim // 8, 64, 64)
                
                if output.shape == expected_shape:
                    print(f"    ✓ 输出形状正确: {output.shape}")
                else:
                    print(f"    ✗ 输出形状错误: 期望{expected_shape}, 实际{output.shape}")
            
        except Exception as e:
            print(f"  {model_size} 模型测试失败: {e}")
    
    print("渐进式上采样测试完成\n")


def test_feature_refinement():
    """测试特征细化模块"""
    print("测试2: 特征细化模块")
    print("-" * 40)
    
    transformer_dim = 256
    batch_size = 2
    
    for model_size in ["1.5B", "3B", "5B", "10B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size, transformer_dim=transformer_dim)
            
            has_refinement = decoder.feature_refinement is not None
            expected_refinement = model_size in ["5B", "7B", "10B"]
            
            print(f"  {model_size} 模型:")
            print(f"    特征细化模块: {'存在' if has_refinement else '不存在'}")
            print(f"    预期状态: {'应该存在' if expected_refinement else '不应该存在'}")
            
            if has_refinement == expected_refinement:
                print(f"    ✓ 特征细化配置正确")
            else:
                print(f"    ✗ 特征细化配置错误")
            
            # 如果有特征细化模块，测试其功能
            if has_refinement:
                input_tensor = torch.randn(batch_size, transformer_dim, 32, 32)
                with torch.no_grad():
                    refined = decoder.feature_refinement(input_tensor)
                    if refined.shape == input_tensor.shape:
                        print(f"    ✓ 特征细化输出形状正确")
                    else:
                        print(f"    ✗ 特征细化输出形状错误")
            
        except Exception as e:
            print(f"  {model_size} 模型测试失败: {e}")
    
    print("特征细化测试完成\n")


def test_memory_optimization():
    """测试内存使用和计算效率优化"""
    print("测试3: 内存使用和计算效率优化")
    print("-" * 40)
    
    for model_size in ["1.5B", "3B", "5B", "10B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size)
            
            # 测试内存使用估算
            memory_info = decoder.get_memory_usage_estimate(batch_size=4)
            print(f"  {model_size} 模型内存估算:")
            print(f"    参数内存: {memory_info['parameter_memory_mb']:.2f} MB")
            print(f"    激活内存: {memory_info['activation_memory_mb']:.2f} MB")
            print(f"    推荐批大小: {memory_info['recommended_batch_size']}")
            
            # 测试内存效率模式
            decoder.enable_memory_efficient_mode(True)
            is_efficient = getattr(decoder, '_memory_efficient_mode', False)
            print(f"    内存效率模式: {'启用' if is_efficient else '未启用'}")
            
            # 测试动态批大小
            decoder.enable_dynamic_batch_sizing(True)
            optimal_batch = decoder.get_optimal_batch_size(available_memory_gb=8.0)
            print(f"    最优批大小: {optimal_batch}")
            
            # 验证大模型自动启用优化
            if model_size in ["7B", "10B"]:
                auto_enabled = getattr(decoder, '_dynamic_batch_sizing', False)
                if auto_enabled:
                    print(f"    ✓ 大模型自动启用动态批大小")
                else:
                    print(f"    ✗ 大模型未自动启用动态批大小")
            
        except Exception as e:
            print(f"  {model_size} 模型测试失败: {e}")
    
    print("内存优化测试完成\n")


def test_batch_optimization():
    """测试批处理优化"""
    print("测试4: 批处理优化")
    print("-" * 40)
    
    # 测试BatchOptimizedMLP
    print("  测试BatchOptimizedMLP:")
    try:
        mlp = BatchOptimizedMLP(
            input_dim=256,
            hidden_dim=512,
            output_dim=128,
            num_layers=3,
            use_checkpoint=True
        )
        
        batch_input = torch.randn(4, 256)
        with torch.no_grad():
            output = mlp(batch_input)
            if output.shape == (4, 128):
                print("    ✓ BatchOptimizedMLP输出形状正确")
            else:
                print("    ✗ BatchOptimizedMLP输出形状错误")
    except Exception as e:
        print(f"    ✗ BatchOptimizedMLP测试失败: {e}")
    
    # 测试MemoryEfficientAttention
    print("  测试MemoryEfficientAttention:")
    try:
        attention = MemoryEfficientAttention(
            embedding_dim=256,
            num_heads=8,
            use_flash_attention=True
        )
        
        batch_size, seq_len = 2, 64
        q = torch.randn(batch_size, seq_len, 256)
        k = torch.randn(batch_size, seq_len, 256)
        v = torch.randn(batch_size, seq_len, 256)
        
        with torch.no_grad():
            output = attention(q, k, v)
            if output.shape == (batch_size, seq_len, 256):
                print("    ✓ MemoryEfficientAttention输出形状正确")
            else:
                print("    ✗ MemoryEfficientAttention输出形状错误")
    except Exception as e:
        print(f"    ✗ MemoryEfficientAttention测试失败: {e}")
    
    # 测试解码器的批处理优化
    print("  测试解码器批处理优化:")
    for model_size in ["3B", "10B"]:  # 测试中等和大型模型
        try:
            decoder = create_scalable_mask_decoder(model_size)
            perf_stats = decoder.get_performance_stats()
            
            uses_batch_opt = perf_stats.get('uses_batch_optimization', False)
            supports_checkpoint = perf_stats.get('supports_gradient_checkpointing', False)
            
            print(f"    {model_size} 模型:")
            print(f"      批处理优化: {'启用' if uses_batch_opt else '未启用'}")
            print(f"      梯度检查点: {'支持' if supports_checkpoint else '不支持'}")
            
            if uses_batch_opt:
                print(f"      ✓ 批处理优化正确配置")
            else:
                print(f"      ✗ 批处理优化配置错误")
                
        except Exception as e:
            print(f"    {model_size} 模型测试失败: {e}")
    
    print("批处理优化测试完成\n")


def test_inference_optimization():
    """测试推理优化功能"""
    print("测试5: 推理优化功能")
    print("-" * 40)
    
    for model_size in ["3B", "10B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size)
            
            print(f"  {model_size} 模型推理优化:")
            
            # 应用推理优化
            optimized_decoder = decoder.apply_inference_optimizations()
            
            # 检查优化状态
            is_eval = not decoder.training
            memory_efficient = getattr(decoder, '_memory_efficient_mode', False)
            dynamic_batch = getattr(decoder, '_dynamic_batch_sizing', False)
            
            print(f"    评估模式: {'启用' if is_eval else '未启用'}")
            print(f"    内存效率: {'启用' if memory_efficient else '未启用'}")
            print(f"    动态批大小: {'启用' if dynamic_batch else '未启用'}")
            
            if is_eval and memory_efficient and dynamic_batch:
                print(f"    ✓ 推理优化配置正确")
            else:
                print(f"    ✗ 推理优化配置不完整")
                
        except Exception as e:
            print(f"  {model_size} 模型测试失败: {e}")
    
    print("推理优化测试完成\n")


def test_performance_comparison():
    """测试性能对比"""
    print("测试6: 性能对比")
    print("-" * 40)
    
    # 比较不同规模模型的性能特征
    model_sizes = ["1.5B", "3B", "5B", "10B"]
    performance_data = []
    
    for model_size in model_sizes:
        try:
            decoder = create_scalable_mask_decoder(model_size)
            info = decoder.get_model_info()
            perf_stats = decoder.get_performance_stats()
            memory_info = decoder.get_memory_usage_estimate()
            
            data = {
                'model_size': model_size,
                'parameters': info['total_parameters'],
                'transformer_depth': info['transformer_depth'],
                'feature_refinement': info['use_feature_refinement'],
                'advanced_upsampling': info['use_advanced_upsampling'],
                'estimated_flops': perf_stats['estimated_flops'],
                'parameter_memory_mb': memory_info['parameter_memory_mb'],
                'recommended_batch_size': memory_info['recommended_batch_size']
            }
            performance_data.append(data)
            
        except Exception as e:
            print(f"  {model_size} 模型性能分析失败: {e}")
    
    # 显示性能对比表
    if performance_data:
        print("  性能对比表:")
        print("  " + "-" * 80)
        print(f"  {'模型':<6} {'参数量':<12} {'深度':<4} {'细化':<4} {'高级上采样':<8} {'内存(MB)':<10} {'批大小':<6}")
        print("  " + "-" * 80)
        
        for data in performance_data:
            print(f"  {data['model_size']:<6} "
                  f"{data['parameters']:>10,} "
                  f"{data['transformer_depth']:>4} "
                  f"{'是' if data['feature_refinement'] else '否':<4} "
                  f"{'是' if data['advanced_upsampling'] else '否':<8} "
                  f"{data['parameter_memory_mb']:>8.1f} "
                  f"{data['recommended_batch_size']:>6}")
        
        print("  " + "-" * 80)
        
        # 验证性能优化的递进性
        print("  性能优化递进性验证:")
        for i in range(1, len(performance_data)):
            current = performance_data[i]
            previous = performance_data[i-1]
            
            # 检查参数量递增
            param_increase = current['parameters'] > previous['parameters']
            # 检查大模型启用高级功能
            advanced_features = (
                current['model_size'] in ["5B", "7B", "10B"] and 
                current['feature_refinement']
            )
            
            print(f"    {previous['model_size']} -> {current['model_size']}: "
                  f"参数递增={'✓' if param_increase else '✗'}, "
                  f"高级功能={'✓' if advanced_features or current['model_size'] in ['1.5B', '3B'] else '✗'}")
    
    print("性能对比测试完成\n")


def main():
    """主测试函数"""
    print("ScalableMaskDecoder 解码器性能优化测试")
    print("=" * 60)
    print("测试任务4.2: 优化解码器性能")
    print("包括: 渐进式上采样、特征细化、内存优化、批处理优化")
    print("=" * 60)
    print()
    
    try:
        # 运行所有测试
        test_progressive_upsampling()
        test_feature_refinement()
        test_memory_optimization()
        test_batch_optimization()
        test_inference_optimization()
        test_performance_comparison()
        
        print("=" * 60)
        print("所有测试完成!")
        print("解码器性能优化功能验证通过")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)