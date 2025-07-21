#!/usr/bin/env python3
"""
CWSAM配置管理系统使用示例

演示如何使用配置管理系统来创建和管理不同规模的CWSAM模型配置。
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from config import (
    CWSAMScalingConfig, 
    ConfigValidator, 
    create_config, 
    validate_config,
    list_model_sizes
)


def demo_basic_usage():
    """演示基本使用方法"""
    print("=== CWSAM配置管理系统基本使用 ===\n")
    
    # 1. 列出所有支持的模型规模
    print("1. 支持的模型规模:")
    sizes = list_model_sizes()
    for size in sizes:
        print(f"   - {size}")
    
    # 2. 创建特定规模的配置
    print("\n2. 创建3B模型配置:")
    config_manager = create_config("3B")
    config_manager.print_config_summary()
    
    # 3. 获取配置详情
    print("\n3. 获取配置字典:")
    config_dict = config_manager.get_config_dict()
    for key, value in config_dict.items():
        print(f"   {key}: {value}")
    
    # 4. 验证配置
    print("\n4. 配置验证:")
    is_valid, errors = validate_config(config_manager.get_config())
    if is_valid:
        print("   ✓ 配置验证通过")
    else:
        print("   ✗ 配置验证失败:")
        for error in errors:
            print(f"     - {error}")


def demo_memory_analysis():
    """演示内存使用分析"""
    print("\n=== 内存使用分析 ===\n")
    
    model_sizes = ["1.5B", "3B", "5B"]
    
    print("不同模型规模的内存使用对比:")
    print(f"{'模型规模':<8} {'参数量(B)':<12} {'FP32推理(GB)':<15} {'FP32训练(GB)':<15} {'FP16推理(GB)':<15} {'FP16训练(GB)':<15}")
    print("-" * 90)
    
    for size in model_sizes:
        config_manager = create_config(size)
        params = config_manager.estimate_parameters() / 1e9
        
        memory_fp32 = config_manager.estimate_memory_usage(batch_size=1, precision="fp32")
        memory_fp16 = config_manager.estimate_memory_usage(batch_size=1, precision="fp16")
        
        print(f"{size:<8} {params:<12.2f} {memory_fp32['total_inference']:<15.2f} "
              f"{memory_fp32['total_training']:<15.2f} {memory_fp16['total_inference']:<15.2f} "
              f"{memory_fp16['total_training']:<15.2f}")


def demo_hardware_compatibility():
    """演示硬件兼容性检查"""
    print("\n=== 硬件兼容性检查 ===\n")
    
    # 测试不同硬件配置
    hardware_configs = [
        ("单卡RTX 4090", 1, 24),
        ("双卡RTX 4090", 2, 24),
        ("4卡RTX 4090", 4, 24),
        ("8卡A100", 8, 80),
    ]
    
    model_size = "5B"
    config = create_config(model_size).get_config()
    
    print(f"测试{model_size}模型在不同硬件配置下的兼容性:")
    
    for name, gpus, memory in hardware_configs:
        print(f"\n{name} ({gpus}x {memory}GB):")
        warnings = ConfigValidator.validate_hardware_compatibility(
            config, available_gpus=gpus, gpu_memory_gb=memory
        )
        
        if not warnings:
            print("   ✓ 硬件兼容性良好")
        else:
            for warning in warnings:
                print(f"   ⚠ {warning}")


def demo_custom_config():
    """演示自定义配置创建"""
    print("\n=== 自定义配置创建 ===\n")
    
    try:
        # 创建自定义配置
        custom_config = CWSAMScalingConfig.create_custom_config(
            model_size="Custom-4B",
            moe_num_experts=96,
            depth=44,
            embed_dim=1792,
            num_heads=28,
            moe_start_layer=18,
            use_multi_scale=True,
            decoder_depth=5
        )
        
        print("自定义配置创建成功:")
        custom_config.print_config_summary()
        
        # 验证自定义配置
        is_valid, errors = validate_config(custom_config.get_config())
        if is_valid:
            print("✓ 自定义配置验证通过")
        else:
            print("✗ 自定义配置验证失败:")
            for error in errors:
                print(f"  - {error}")
                
    except Exception as e:
        print(f"创建自定义配置失败: {e}")


def demo_config_comparison():
    """演示配置对比"""
    print("\n=== 配置对比分析 ===\n")
    
    sizes_to_compare = ["1.5B", "3B", "5B"]
    
    print("模型配置对比:")
    print(f"{'规模':<6} {'专家数':<8} {'深度':<6} {'嵌入维度':<10} {'注意力头':<10} {'MoE起始层':<12} {'参数量(B)':<12}")
    print("-" * 80)
    
    for size in sizes_to_compare:
        config = create_config(size).get_config()
        params = config.estimate_parameters() / 1e9
        
        print(f"{size:<6} {config.moe_num_experts:<8} {config.depth:<6} "
              f"{config.embed_dim:<10} {config.num_heads:<10} "
              f"{config.moe_start_layer:<12} {params:<12.2f}")


def main():
    """主演示函数"""
    print("CWSAM模型参数扩展配置管理系统演示")
    print("=" * 60)
    
    try:
        demo_basic_usage()
        demo_memory_analysis()
        demo_hardware_compatibility()
        demo_custom_config()
        demo_config_comparison()
        
        print("\n" + "=" * 60)
        print("✓ 演示完成")
        print("\n使用建议:")
        print("1. 根据可用硬件选择合适的模型规模")
        print("2. 使用FP16精度可以显著减少内存使用")
        print("3. 大模型建议使用多GPU训练")
        print("4. 在生产环境中使用前请验证配置")
        
    except Exception as e:
        print(f"\n✗ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()