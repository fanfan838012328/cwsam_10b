#!/usr/bin/env python3
"""
测试CWSAM配置管理系统
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入配置模块，避免导入整个models包
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

try:
    from config import (
        CWSAMScalingConfig, 
        ConfigValidator, 
        create_config, 
        validate_config,
        list_model_sizes
    )
except ImportError as e:
    print(f"导入错误: {e}")
    print("尝试直接运行配置模块...")
    import config
    CWSAMScalingConfig = config.CWSAMScalingConfig
    ConfigValidator = config.ConfigValidator
    create_config = config.create_config
    validate_config = config.validate_config
    list_model_sizes = config.list_model_sizes


def test_predefined_configs():
    """测试预定义配置"""
    print("=== 测试预定义配置 ===")
    
    sizes = list_model_sizes()
    print(f"支持的模型规模: {sizes}")
    
    for size in sizes:
        try:
            config_manager = create_config(size)
            config = config_manager.get_config()
            
            print(f"\n{size} 模型配置:")
            print(f"  专家数量: {config.moe_num_experts}")
            print(f"  深度: {config.depth}")
            print(f"  嵌入维度: {config.embed_dim}")
            print(f"  注意力头数: {config.num_heads}")
            print(f"  预估参数量: {config.estimate_parameters():,}")
            
            # 验证配置
            is_valid, errors = validate_config(config)
            if is_valid:
                print(f"  ✓ 配置验证通过")
            else:
                print(f"  ✗ 配置验证失败: {errors}")
                
        except Exception as e:
            print(f"  ✗ 创建{size}配置失败: {e}")


def test_parameter_estimation():
    """测试参数量估算"""
    print("\n=== 测试参数量估算 ===")
    
    for size in ["1.5B", "3B", "5B", "10B"]:
        config_manager = create_config(size)
        params = config_manager.estimate_parameters()
        target_params = float(size.replace("B", "")) * 1e9
        
        print(f"{size}: 估算 {params:,} 参数 ({params/1e9:.2f}B)")
        print(f"  目标: {target_params/1e9:.1f}B, 差异: {abs(params - target_params)/target_params*100:.1f}%")


def test_memory_estimation():
    """测试内存使用估算"""
    print("\n=== 测试内存使用估算 ===")
    
    config_manager = create_config("3B")
    memory_fp32 = config_manager.estimate_memory_usage(batch_size=1, precision="fp32")
    memory_fp16 = config_manager.estimate_memory_usage(batch_size=1, precision="fp16")
    
    print("3B模型内存使用估算:")
    print(f"  FP32 推理: {memory_fp32['total_inference']:.2f} GB")
    print(f"  FP32 训练: {memory_fp32['total_training']:.2f} GB")
    print(f"  FP16 推理: {memory_fp16['total_inference']:.2f} GB")
    print(f"  FP16 训练: {memory_fp16['total_training']:.2f} GB")


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")
    
    # 测试有效配置
    valid_config = create_config("3B").get_config()
    is_valid, errors = validate_config(valid_config)
    print(f"有效配置验证: {'通过' if is_valid else '失败'}")
    if errors:
        print(f"  错误: {errors}")
    
    # 测试无效配置
    try:
        invalid_config_manager = CWSAMScalingConfig.create_custom_config(
            model_size="test",
            moe_num_experts=3,  # 太少
            depth=8,  # 太浅
            embed_dim=100,  # 太小
            num_heads=7,  # 不能整除embed_dim
            moe_start_layer=10  # 超过depth
        )
        
        is_valid, errors = validate_config(invalid_config_manager.get_config())
        print(f"无效配置验证: {'通过' if is_valid else '失败'}")
        print(f"  预期错误数量: {len(errors)}")
        for i, error in enumerate(errors[:3]):  # 只显示前3个错误
            print(f"    {i+1}. {error}")
        if len(errors) > 3:
            print(f"    ... 还有 {len(errors)-3} 个错误")
            
    except Exception as e:
        print(f"创建无效配置时出错: {e}")


def test_hardware_compatibility():
    """测试硬件兼容性检查"""
    print("\n=== 测试硬件兼容性检查 ===")
    
    config = create_config("10B").get_config()
    
    # 测试不同硬件配置
    hardware_configs = [
        (1, 24),  # 1x RTX 4090
        (4, 24),  # 4x RTX 4090
        (8, 80),  # 8x A100
    ]
    
    for gpus, memory in hardware_configs:
        warnings = ConfigValidator.validate_hardware_compatibility(
            config, available_gpus=gpus, gpu_memory_gb=memory
        )
        print(f"{gpus}x {memory}GB GPU:")
        if warnings:
            for warning in warnings:
                print(f"  ⚠ {warning}")
        else:
            print("  ✓ 硬件兼容性良好")


def main():
    """主测试函数"""
    print("CWSAM配置管理系统测试")
    print("=" * 50)
    
    try:
        test_predefined_configs()
        test_parameter_estimation()
        test_memory_estimation()
        test_config_validation()
        test_hardware_compatibility()
        
        print("\n" + "=" * 50)
        print("✓ 所有测试完成")
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()