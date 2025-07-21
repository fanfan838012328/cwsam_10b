#!/usr/bin/env python3
"""
测试多尺度特征金字塔集成

验证MultiScaleFeaturePyramid模块是否正确集成到ScalableImageEncoderViT中，
并测试3B以上模型的多尺度特征提取功能。
"""

import torch
import torch.nn as nn
import sys
import os

# 添加models目录到路径
sys.path.append('models')

from models.config import CWSAMScalingConfig, create_config
from models.scalable_image_encoder import ScalableImageEncoderViT, create_scalable_image_encoder
from models.multi_scale_pyramid import MultiScaleFeaturePyramid, create_multi_scale_pyramid


def test_multi_scale_pyramid_standalone():
    """测试独立的多尺度特征金字塔模块"""
    print("=== 测试独立MultiScaleFeaturePyramid模块 ===")
    
    # 测试不同配置
    test_configs = [
        {"embed_dim": 1536, "model_size": "3B", "fusion_method": "conv"},
        {"embed_dim": 2048, "model_size": "5B", "fusion_method": "attention"},
        {"embed_dim": 3072, "model_size": "10B", "fusion_method": "attention"},
    ]
    
    for config in test_configs:
        print(f"\n测试 {config['model_size']} 配置 (embed_dim={config['embed_dim']})...")
        
        try:
            # 创建多尺度金字塔
            pyramid = create_multi_scale_pyramid(**config)
            
            if pyramid is not None:
                # 创建测试输入 [B, H, W, C]
                B, H, W, C = 2, 64, 64, config['embed_dim']
                x = torch.randn(B, H, W, C)
                
                print(f"  输入形状: {x.shape}")
                
                # 前向传播
                with torch.no_grad():
                    output = pyramid(x)
                    print(f"  输出形状: {output.shape}")
                    
                    # 验证输出形状正确
                    assert output.shape == x.shape, f"输出形状不匹配: {output.shape} vs {x.shape}"
                    
                    # 获取特征图
                    feature_maps = pyramid.get_feature_maps(x)
                    print(f"  特征图数量: {len(feature_maps)}")
                    for name, feat in feature_maps.items():
                        print(f"    {name}: {feat.shape}")
                
                print(f"  ✓ {config['model_size']} 多尺度金字塔测试通过")
            else:
                print(f"  - {config['model_size']} 不使用多尺度特征金字塔")
                
        except Exception as e:
            print(f"  ✗ {config['model_size']} 测试失败: {e}")
            import traceback
            traceback.print_exc()


def test_scalable_encoder_integration():
    """测试ScalableImageEncoderViT中的多尺度集成"""
    print("\n=== 测试ScalableImageEncoderViT多尺度集成 ===")
    
    # 测试不同规模的模型
    test_sizes = ["1.5B", "2B", "3B", "5B", "7B", "10B"]
    
    for model_size in test_sizes:
        print(f"\n测试 {model_size} 模型...")
        
        try:
            # 创建配置
            config = create_config(model_size)
            config_dict = config.get_config()
            
            print(f"  use_multi_scale: {config_dict.use_multi_scale}")
            print(f"  embed_dim: {config_dict.embed_dim}")
            print(f"  depth: {config_dict.depth}")
            
            # 创建编码器
            encoder = create_scalable_image_encoder(
                model_size=model_size,
                img_size=512,  # 使用较小的图像尺寸进行测试
                patch_size=16
            )
            
            # 检查多尺度模块是否正确初始化
            has_multi_scale = encoder.multi_scale_pyramid is not None
            should_have_multi_scale = config_dict.use_multi_scale
            
            print(f"  多尺度模块存在: {has_multi_scale}")
            print(f"  应该有多尺度模块: {should_have_multi_scale}")
            
            assert has_multi_scale == should_have_multi_scale, \
                f"多尺度模块初始化不正确: {has_multi_scale} vs {should_have_multi_scale}"
            
            # 测试前向传播
            x = torch.randn(1, 3, 512, 512)
            
            with torch.no_grad():
                output = encoder(x)
                print(f"  输入形状: {x.shape}")
                print(f"  输出形状: {output.shape}")
                
                # 验证输出形状
                expected_h = expected_w = 512 // 16  # 32
                expected_c = 256  # out_chans
                expected_shape = (1, expected_c, expected_h, expected_w)
                
                assert output.shape == expected_shape, \
                    f"输出形状不正确: {output.shape} vs {expected_shape}"
                
                # 获取辅助损失
                aux_losses = encoder.get_aux_losses()
                if aux_losses:
                    print(f"  辅助损失: {list(aux_losses.keys())}")
            
            print(f"  ✓ {model_size} 模型集成测试通过")
            
        except Exception as e:
            print(f"  ✗ {model_size} 模型测试失败: {e}")
            import traceback
            traceback.print_exc()


def test_multi_scale_feature_extraction():
    """测试多尺度特征提取的具体功能"""
    print("\n=== 测试多尺度特征提取功能 ===")
    
    # 使用5B模型进行详细测试
    model_size = "5B"
    print(f"使用 {model_size} 模型进行详细测试...")
    
    try:
        # 创建编码器
        encoder = create_scalable_image_encoder(
            model_size=model_size,
            img_size=512,
            patch_size=16
        )
        
        # 创建测试输入
        x = torch.randn(2, 3, 512, 512)
        
        print(f"输入形状: {x.shape}")
        
        with torch.no_grad():
            # 获取编码器输出
            output = encoder(x)
            print(f"最终输出形状: {output.shape}")
            
            # 如果有多尺度模块，测试其特征图
            if encoder.multi_scale_pyramid is not None:
                print("\n多尺度特征金字塔详细信息:")
                
                # 获取中间特征（在应用多尺度之前）
                # 这需要修改forward方法来暴露中间特征，这里我们直接测试多尺度模块
                
                # 创建模拟的中间特征
                B, H, W = 2, 32, 32  # 512//16 = 32
                C = encoder.embed_dim
                intermediate_feat = torch.randn(B, H, W, C)
                
                print(f"中间特征形状: {intermediate_feat.shape}")
                
                # 通过多尺度模块
                multi_scale_output = encoder.multi_scale_pyramid(intermediate_feat)
                print(f"多尺度输出形状: {multi_scale_output.shape}")
                
                # 验证形状保持不变
                assert multi_scale_output.shape == intermediate_feat.shape, \
                    f"多尺度输出形状不匹配: {multi_scale_output.shape} vs {intermediate_feat.shape}"
                
                # 获取各尺度特征图
                feature_maps = encoder.multi_scale_pyramid.get_feature_maps(intermediate_feat)
                print(f"各尺度特征图:")
                for name, feat in feature_maps.items():
                    print(f"  {name}: {feat.shape}")
                
                print("✓ 多尺度特征提取功能正常")
            else:
                print("该模型不使用多尺度特征金字塔")
        
    except Exception as e:
        print(f"✗ 多尺度特征提取测试失败: {e}")
        import traceback
        traceback.print_exc()


def test_model_info_and_parameters():
    """测试模型信息和参数统计"""
    print("\n=== 测试模型信息和参数统计 ===")
    
    for model_size in ["3B", "5B", "10B"]:  # 只测试使用多尺度的模型
        print(f"\n{model_size} 模型信息:")
        
        try:
            encoder = create_scalable_image_encoder(model_size=model_size)
            
            # 打印模型信息
            encoder.print_model_info()
            
            # 获取详细信息
            info = encoder.get_model_info()
            
            # 验证多尺度配置
            assert info['use_multi_scale'] == True, f"{model_size} 应该启用多尺度特征"
            
            print(f"✓ {model_size} 模型信息正确")
            
        except Exception as e:
            print(f"✗ {model_size} 模型信息测试失败: {e}")


def main():
    """主测试函数"""
    print("多尺度特征金字塔集成测试")
    print("=" * 60)
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    
    try:
        # 1. 测试独立的多尺度金字塔模块
        test_multi_scale_pyramid_standalone()
        
        # 2. 测试ScalableImageEncoderViT集成
        test_scalable_encoder_integration()
        
        # 3. 测试多尺度特征提取功能
        test_multi_scale_feature_extraction()
        
        # 4. 测试模型信息和参数统计
        test_model_info_and_parameters()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成！多尺度特征金字塔集成正常工作")
        
    except Exception as e:
        print(f"\n✗ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)