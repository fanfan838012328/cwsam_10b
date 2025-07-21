"""
测试内存使用情况的脚本
"""

import torch
import yaml
import models
from models.model_factory import create_model, print_model_info
import gc

def test_memory_usage():
    """测试不同配置下的内存使用情况"""
    
    # 清理初始内存
    torch.cuda.empty_cache()
    gc.collect()
    
    print("=== 内存使用测试 ===")
    print(f"GPU总内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 测试配置
    configs = [
        {
            "name": "1.5B优化版",
            "config_path": "configs/train/sam/train_sam_moe_1dot5b_optimized.yaml"
        },
        {
            "name": "3B优化版", 
            "config_path": "configs/train/sam/train_sam_moe_3b_optimized.yaml"
        }
    ]
    
    for config_info in configs:
        print(f"\n--- 测试 {config_info['name']} ---")
        
        try:
            # 加载配置
            with open(config_info['config_path'], 'r') as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
            
            # 创建模型
            model = models.make(config['model'])
            model = model.cuda()
            
            # 测试前向传播
            inp_size = config['model']['args']['inp_size']
            batch_size = config['train_dataset']['batch_size']
            
            print(f"输入尺寸: {inp_size}x{inp_size}")
            print(f"批次大小: {batch_size}")
            
            # 创建测试输入
            test_input = torch.randn(batch_size, 3, inp_size, inp_size).cuda()
            test_gt = torch.randn(batch_size, 33, inp_size, inp_size).cuda()
            
            # 记录内存使用
            torch.cuda.synchronize()
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"模型加载后内存: {memory_before:.2f} GB")
            
            # 前向传播测试
            with torch.no_grad():
                model.set_input(test_input, test_gt)
                output = model.forward()
                
                torch.cuda.synchronize()
                memory_after = torch.cuda.memory_allocated() / 1024**3
                print(f"前向传播后内存: {memory_after:.2f} GB")
                print(f"前向传播内存增长: {memory_after - memory_before:.2f} GB")
                print(f"输出形状: {output.shape}")
            
            # 清理
            del model, test_input, test_gt, output
            torch.cuda.empty_cache()
            gc.collect()
            
            print("✓ 测试通过")
            
        except Exception as e:
            print(f"✗ 测试失败: {e}")
            # 清理
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_memory_usage()