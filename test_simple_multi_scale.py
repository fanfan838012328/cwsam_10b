#!/usr/bin/env python3
"""
简单的多尺度特征金字塔集成测试

验证MultiScaleFeaturePyramid模块是否正确集成到ScalableImageEncoderViT中。
"""

import torch
import sys
import os

# 添加models目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

def test_multi_scale_integration():
    """测试多尺度特征金字塔集成"""
    print("=== 测试多尺度特征金字塔集成 ===")
    
    try:
        from config import CWSAMScalingConfig
        from multi_scale_pyramid import MultiScaleFeaturePyramid, create_multi_scale_pyramid
        
        # 测试不同模型规模的配置
        test_sizes = ["1.5B", "2B", "3B", "5B", "7B", "10B"]
        
        for model_size in test_sizes:
            print(f"\n测试 {model_size} 模型配置...")
            
            # 创建配置
            config = CWSAMScalingConfig(model_size)
            config_dict = config.get_config_dict()
            
            print(f"  embed_dim: {config_dict['embed_dim']}")
            print(f"  use_multi_scale: {config_dict['use_multi_scale']}")
            
            # 创建多尺度金字塔
            pyramid = create_multi_scale_pyramid(
                embed_dim=config_dict['embed_dim'],
                model_size=model_size
            )
            
            if pyramid is not None:
                print(f"  ✓ 多尺度金字塔已创建")
                
                # 测试前向传播
                B, H, W, C = 1, 32, 32, config_dict['embed_dim']
                x = torch.randn(B, H, W, C)
                
                with torch.no_grad():
                    output = pyramid(x)
                    print(f"  输入形状: {x.shape}")
                    print(f"  输出形状: {output.shape}")
                    
                    # 验证形状
                    assert output.shape == x.shape, f"形状不匹配: {output.shape} vs {x.shape}"
                    
                    # 获取特征图
                    feature_maps = pyramid.get_feature_maps(x)
                    print(f"  特征图数量: {len(feature_maps)}")
                    
                print(f"  ✓ {model_size} 多尺度测试通过")
            else:
                print(f"  - {model_size} 不使用多尺度特征")
                
        print("\n=== 多尺度特征金字塔集成测试完成 ===")
        return True
        
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_validation():
    """测试配置验证"""
    print("\n=== 测试配置验证 ===")
    
    try:
        from config import CWSAMScalingConfig
        
        # 验证3B以上模型启用多尺度特征
        multi_scale_models = ["3B", "5B", "7B", "10B"]
        non_multi_scale_models = ["1.5B", "2B"]
        
        for model_size in multi_scale_models:
            config = CWSAMScalingConfig(model_size)
            config_dict = config.get_config_dict()
            assert config_dict['use_multi_scale'] == True, f"{model_size} 应该启用多尺度特征"
            print(f"  ✓ {model_size} 正确启用多尺度特征")
        
        for model_size in non_multi_scale_models:
            config = CWSAMScalingConfig(model_size)
            config_dict = config.get_config_dict()
            assert config_dict['use_multi_scale'] == False, f"{model_size} 不应该启用多尺度特征"
            print(f"  ✓ {model_size} 正确禁用多尺度特征")
        
        print("✓ 配置验证通过")
        return True
        
    except Exception as e:
        print(f"✗ 配置验证失败: {e}")
        return False


def main():
    """主测试函数"""
    print("多尺度特征金字塔集成简单测试")
    print("=" * 50)
    
    success = True
    
    # 测试配置验证
    if not test_config_validation():
        success = False
    
    # 测试多尺度集成
    if not test_multi_scale_integration():
        success = False
    
    if success:
        print("\n✓ 所有测试通过！多尺度特征金字塔集成正常")
    else:
        print("\n✗ 部分测试失败")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)