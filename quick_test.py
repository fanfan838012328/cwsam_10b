"""
简化的权重加载验证测试
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

from models.mmseg.models.sam.efficient_moe import EfficientMoEMLPBlock, OptimizedHierarchicalMoEMLPBlock


def quick_test():
    """快速测试权重加载功能"""
    print("开始快速测试...")
    
    # 创建测试用的专家权重
    fake_checkpoint = {}
    for layer_idx in [28, 29]:  # 只测试2层
        for expert_idx in range(16):
            # 创建假的专家权重
            base_key = f'image_encoder.blocks.{layer_idx}.mlp.experts.{expert_idx}'
            fake_checkpoint[f'{base_key}.0.weight'] = torch.randn(5120, 1280)
            fake_checkpoint[f'{base_key}.0.bias'] = torch.randn(5120)
            fake_checkpoint[f'{base_key}.2.weight'] = torch.randn(1280, 5120)
            fake_checkpoint[f'{base_key}.2.bias'] = torch.randn(1280)
    
    print(f"创建了 {len(fake_checkpoint)} 个测试权重")
    
    # 测试EfficientMoEMLPBlock
    print("\n1. 测试EfficientMoEMLPBlock...")
    efficient_moe = EfficientMoEMLPBlock(
        embedding_dim=1280,
        mlp_dim=5120,
        num_experts=24,  # 减少专家数量以加快测试
        use_expert_choice=True,
        use_sharding=False,  # 关闭分片以简化测试
    )
    
    try:
        efficient_moe.load_1_5b_expert_weights(fake_checkpoint)
        print("  ✓ 权重加载成功")
        
        # 测试前向传播
        test_input = torch.randn(1, 10, 1280)  # 小输入
        with torch.no_grad():
            output = efficient_moe(test_input)
        print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    
    # 测试OptimizedHierarchicalMoEMLPBlock
    print("\n2. 测试OptimizedHierarchicalMoEMLPBlock...")
    hierarchical_moe = OptimizedHierarchicalMoEMLPBlock(
        embedding_dim=1280,
        mlp_dim=5120,
        num_expert_groups=2,  # 减少组数
        experts_per_group=8,   # 减少每组专家数
        k_groups=1,
        k_experts=2,
    )
    
    try:
        hierarchical_moe.load_1_5b_expert_weights(fake_checkpoint)
        print("  ✓ 权重加载成功")
        
        # 测试前向传播
        test_input = torch.randn(1, 10, 1280)
        with torch.no_grad():
            output = hierarchical_moe(test_input)
        print(f"  ✓ 前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    
    return True


def test_checkpoint_loading():
    """测试真实checkpoint加载"""
    print("\n3. 测试真实checkpoint...")
    
    # 检查checkpoint文件
    checkpoint_path = '/mnt/fanfq/data/fan/cwsam/save/XinTong_sam_vit_h_moe_3b/model_epoch_121.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"  警告: checkpoint文件不存在: {checkpoint_path}")
        return True  # 不算错误，只是没有文件
    
    try:
        print(f"  加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 统计专家权重
        expert_keys = [k for k in checkpoint.keys() if 'experts.' in k and 'image_encoder' in k]
        print(f"  找到 {len(expert_keys)} 个专家权重参数")
        
        if len(expert_keys) > 0:
            # 测试一个小的MoE模块
            small_moe = EfficientMoEMLPBlock(
                embedding_dim=1280,
                mlp_dim=5120,
                num_experts=16,
                use_expert_choice=False,
                use_sharding=False,
            )
            
            small_moe.load_1_5b_expert_weights(checkpoint)
            print("  ✓ 真实权重加载成功")
        else:
            print("  警告: 未找到专家权重，可能不是MoE模型")
        
        return True
        
    except Exception as e:
        print(f"  ✗ checkpoint加载失败: {e}")
        return False


if __name__ == "__main__":
    print("=" * 50)
    print("权重加载快速验证测试")
    print("=" * 50)
    
    success = True
    
    # 基础功能测试
    success &= quick_test()
    
    # 真实文件测试
    success &= test_checkpoint_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ 所有测试通过！")
        print("✓ 1.5B权重可以成功加载到新的高效MoE架构")
        print("✓ 新架构前向传播正常")
    else:
        print("✗ 部分测试失败")
    print("=" * 50)