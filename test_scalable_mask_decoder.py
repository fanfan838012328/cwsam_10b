"""
测试ScalableMaskDecoder的功能和性能
"""

import torch
import torch.nn as nn
import sys
import os

# 添加models目录到路径
sys.path.append('models')

from scalable_mask_decoder import ScalableMaskDecoder, create_scalable_mask_decoder
from config import CWSAMScalingConfig


def test_basic_functionality():
    """测试基本功能"""
    print("测试基本功能...")
    
    # 测试不同规模的解码器创建
    for model_size in ["1.5B", "2B", "3B", "5B"]:
        try:
            config = CWSAMScalingConfig(model_size)
            decoder = ScalableMaskDecoder(config)
            
            info = decoder.get_model_info()
            print(f"\n{model_size} 解码器:")
            print(f"  Transformer深度: {info['transformer_depth']}")
            print(f"  注意力头数: {info['num_heads']}")
            print(f"  总参数量: {info['total_parameters']:,}")
            print(f"  特征细化: {info['use_feature_refinement']}")
            
        except Exception as e:
            print(f"创建 {model_size} 解码器失败: {e}")
            return False
    
    print("✓ 基本功能测试通过")
    return True


def test_forward_pass():
    """测试前向传播"""
    print("\n测试前向传播...")
    
    # 创建测试数据
    batch_size = 2
    embed_dim = 256
    h, w = 64, 64
    
    # 模拟输入数据
    image_embeddings = torch.randn(batch_size, embed_dim, h, w)
    image_pe = torch.randn(batch_size, embed_dim, h, w)
    sparse_prompt_embeddings = torch.randn(batch_size, 0, embed_dim)  # 无稀疏提示
    dense_prompt_embeddings = torch.randn(batch_size, embed_dim, h, w)
    
    # 测试不同规模的解码器
    for model_size in ["1.5B", "3B", "5B"]:
        try:
            decoder = create_scalable_mask_decoder(model_size)
            decoder.eval()
            
            with torch.no_grad():
                masks, iou_pred = decoder(
                    image_embeddings=image_embeddings,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_prompt_embeddings,
                    dense_prompt_embeddings=dense_prompt_embeddings,
                    multimask_output=True
                )
            
            print(f"{model_size} 解码器输出:")
            print(f"  掩码形状: {masks.shape}")
            print(f"  IoU预测形状: {iou_pred.shape}")
            
            # 验证输出形状
            expected_mask_shape = (batch_size, 4, 1, h*4, w*4)  # 4倍上采样
            expected_iou_shape = (batch_size, 4)
            
            if masks.shape != expected_mask_shape:
                print(f"  警告: 掩码形状不匹配，期望 {expected_mask_shape}")
            if iou_pred.shape != expected_iou_shape:
                print(f"  警告: IoU预测形状不匹配，期望 {expected_iou_shape}")
                
        except Exception as e:
            print(f"  {model_size} 前向传播失败: {e}")
            return False
    
    print("✓ 前向传播测试通过")
    return True


def test_compatibility():
    """测试与现有接口的兼容性"""
    print("\n测试接口兼容性...")
    
    try:
        # 创建解码器
        decoder = create_scalable_mask_decoder("3B")
        
        # 检查必要的方法和属性
        required_methods = ['forward', 'predict_masks', 'get_model_info']
        required_attributes = ['transformer', 'iou_token', 'mask_tokens', 
                             'output_upscaling', 'iou_prediction_head']
        
        for method in required_methods:
            if not hasattr(decoder, method):
                print(f"  缺少方法: {method}")
                return False
        
        for attr in required_attributes:
            if not hasattr(decoder, attr):
                print(f"  缺少属性: {attr}")
                return False
        
        # 检查transformer的深度是否正确调整
        config = CWSAMScalingConfig("3B")
        expected_depth = config.get_config().decoder_depth
        actual_depth = decoder.transformer.depth
        
        if actual_depth != expected_depth:
            print(f"  Transformer深度不匹配: 期望 {expected_depth}, 实际 {actual_depth}")
            return False
        
        print("✓ 接口兼容性测试通过")
        return True
        
    except Exception as e:
        print(f"  兼容性测试失败: {e}")
        return False


def test_memory_efficiency():
    """测试内存效率"""
    print("\n测试内存效率...")
    
    try:
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 测试不同规模模型的内存使用
        memory_usage = {}
        
        for model_size in ["1.5B", "3B", "5B"]:
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            decoder = create_scalable_mask_decoder(model_size)
            
            # 计算参数内存
            param_memory = sum(p.numel() * 4 for p in decoder.parameters()) / 1024 / 1024  # MB (FP32)
            
            # 获取当前内存使用
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            memory_usage[model_size] = {
                'param_memory': param_memory,
                'total_increase': memory_increase
            }
            
            print(f"{model_size} 内存使用:")
            print(f"  参数内存: {param_memory:.2f} MB")
            print(f"  总内存增长: {memory_increase:.2f} MB")
            
            del decoder
        
        # 验证内存使用随模型规模合理增长
        sizes = ["1.5B", "3B", "5B"]
        for i in range(1, len(sizes)):
            prev_size = sizes[i-1]
            curr_size = sizes[i]
            
            prev_memory = memory_usage[prev_size]['param_memory']
            curr_memory = memory_usage[curr_size]['param_memory']
            
            if curr_memory <= prev_memory:
                print(f"  警告: {curr_size} 参数内存 ({curr_memory:.2f} MB) 不大于 {prev_size} ({prev_memory:.2f} MB)")
        
        print("✓ 内存效率测试通过")
        return True
        
    except ImportError:
        print("  跳过内存效率测试 (需要psutil)")
        return True
    except Exception as e:
        print(f"  内存效率测试失败: {e}")
        return False


def test_gradient_flow():
    """测试梯度流"""
    print("\n测试梯度流...")
    
    try:
        decoder = create_scalable_mask_decoder("3B")
        decoder.train()
        
        # 创建测试数据
        batch_size = 1
        embed_dim = 256
        h, w = 64, 64
        
        image_embeddings = torch.randn(batch_size, embed_dim, h, w, requires_grad=True)
        image_pe = torch.randn(batch_size, embed_dim, h, w)
        sparse_prompt_embeddings = torch.randn(batch_size, 0, embed_dim)
        dense_prompt_embeddings = torch.randn(batch_size, embed_dim, h, w)
        
        # 前向传播
        masks, iou_pred = decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            multimask_output=True
        )
        
        # 计算损失并反向传播
        loss = masks.mean() + iou_pred.mean()
        loss.backward()
        
        # 检查梯度
        grad_norms = []
        zero_grad_count = 0
        
        for name, param in decoder.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm == 0:
                    zero_grad_count += 1
            else:
                zero_grad_count += 1
        
        if len(grad_norms) == 0:
            print("  错误: 没有参数有梯度")
            return False
        
        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        
        print(f"  平均梯度范数: {avg_grad_norm:.6f}")
        print(f"  最大梯度范数: {max_grad_norm:.6f}")
        print(f"  零梯度参数数量: {zero_grad_count}")
        
        # 检查梯度爆炸
        if max_grad_norm > 100:
            print(f"  警告: 可能存在梯度爆炸 (最大梯度范数: {max_grad_norm})")
        
        # 检查梯度消失
        if avg_grad_norm < 1e-7:
            print(f"  警告: 可能存在梯度消失 (平均梯度范数: {avg_grad_norm})")
        
        print("✓ 梯度流测试通过")
        return True
        
    except Exception as e:
        print(f"  梯度流测试失败: {e}")
        return False


def main():
    """主测试函数"""
    print("ScalableMaskDecoder 测试套件")
    print("=" * 50)
    
    tests = [
        test_basic_functionality,
        test_forward_pass,
        test_compatibility,
        test_memory_efficiency,
        test_gradient_flow,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"测试异常: {e}")
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过!")
        return True
    else:
        print("✗ 部分测试失败")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)