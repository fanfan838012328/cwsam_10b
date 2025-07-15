"""
验证1.5B权重在新高效MoE架构中的加载正确性
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

import models
from models.mmseg.models.sam.efficient_moe import EfficientMoEMLPBlock, OptimizedHierarchicalMoEMLPBlock


def test_weight_loading():
    """测试权重加载功能"""
    print("开始测试1.5B权重加载到高效MoE架构...")
    
    # 1. 加载配置
    config_path = '/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 2. 创建模型
    print("创建10B模型...")
    model = models.make(config['model'])
    
    # 3. 加载1.5B checkpoint
    checkpoint_path = config.get('sam_checkpoint')
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"错误: 找不到checkpoint文件 {checkpoint_path}")
        return False
    
    print(f"加载1.5B checkpoint: {checkpoint_path}")
    sam_checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 4. 统计原始专家权重
    original_expert_count = 0
    for key in sam_checkpoint.keys():
        if 'experts.' in key and 'image_encoder' in key:
            original_expert_count += 1
    
    print(f"找到 {original_expert_count} 个原始专家权重参数")
    
    # 5. 应用权重过滤加载
    model_dict = model.state_dict()
    filtered_state_dict = {
        k: v for k, v in sam_checkpoint.items()
        if k in model_dict and model_dict[k].shape == v.shape
    }
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # 6. 初始化高效MoE权重
    print("初始化高效MoE权重...")
    efficient_moe_layers = []
    for name, module in model.named_modules():
        if hasattr(module, 'load_1_5b_expert_weights'):
            efficient_moe_layers.append((name, module))
            print(f"找到高效MoE层: {name}")
    
    if len(efficient_moe_layers) == 0:
        print("警告: 未找到高效MoE层")
        return False
    
    # 7. 测试权重加载
    success_count = 0
    for layer_name, moe_module in efficient_moe_layers:
        try:
            print(f"测试层 {layer_name} 的权重加载...")
            moe_module.load_1_5b_expert_weights(sam_checkpoint)
            success_count += 1
            print(f"✓ 层 {layer_name} 权重加载成功")
        except Exception as e:
            print(f"✗ 层 {layer_name} 权重加载失败: {e}")
    
    # 8. 验证参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n权重加载统计:")
    print(f"成功加载的MoE层: {success_count}/{len(efficient_moe_layers)}")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"冻结参数: {frozen_params:,}")
    print(f"参数量级: {total_params / 1e9:.1f}B")
    
    # 9. 验证模型可以前向传播
    print("\n测试模型前向传播...")
    try:
        model.eval()
        with torch.no_grad():
            # 创建dummy输入
            dummy_input = torch.randn(1, 3, 512, 512)
            dummy_gt = torch.zeros(1, 33, 512, 512)
            dummy_gt[:, 0] = 1  # 背景类
            
            model.set_input(dummy_input, dummy_gt)
            output = model.infer(dummy_input)
            
            print(f"✓ 前向传播成功，输出形状: {output.shape}")
            return True
            
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        return False


def test_specific_moe_blocks():
    """测试特定的MoE模块"""
    print("\n测试特定MoE模块...")
    
    # 创建测试用的专家权重
    fake_checkpoint = {}
    for layer_idx in [28, 29, 30, 31]:  # 假设后4层有MoE
        for expert_idx in range(16):
            # 创建假的专家权重
            base_key = f'image_encoder.blocks.{layer_idx}.mlp.experts.{expert_idx}'
            fake_checkpoint[f'{base_key}.0.weight'] = torch.randn(5120, 1280)
            fake_checkpoint[f'{base_key}.0.bias'] = torch.randn(5120)
            fake_checkpoint[f'{base_key}.2.weight'] = torch.randn(1280, 5120)
            fake_checkpoint[f'{base_key}.2.bias'] = torch.randn(1280)
    
    print(f"创建了 {len(fake_checkpoint)} 个假权重用于测试")
    
    # 测试EfficientMoEMLPBlock
    print("\n测试EfficientMoEMLPBlock...")
    efficient_moe = EfficientMoEMLPBlock(
        embedding_dim=1280,
        mlp_dim=5120,
        num_experts=48,
        use_expert_choice=True,
        use_sharding=True,
    )
    
    try:
        efficient_moe.load_1_5b_expert_weights(fake_checkpoint)
        print("✓ EfficientMoEMLPBlock权重加载成功")
        
        # 测试前向传播
        test_input = torch.randn(2, 100, 1280)
        output = efficient_moe(test_input)
        print(f"✓ EfficientMoEMLPBlock前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"✗ EfficientMoEMLPBlock测试失败: {e}")
    
    # 测试OptimizedHierarchicalMoEMLPBlock
    print("\n测试OptimizedHierarchicalMoEMLPBlock...")
    hierarchical_moe = OptimizedHierarchicalMoEMLPBlock(
        embedding_dim=1280,
        mlp_dim=5120,
        num_expert_groups=3,
        experts_per_group=16,
        k_groups=1,
        k_experts=4,
    )
    
    try:
        hierarchical_moe.load_1_5b_expert_weights(fake_checkpoint)
        print("✓ OptimizedHierarchicalMoEMLPBlock权重加载成功")
        
        # 测试前向传播
        test_input = torch.randn(2, 100, 1280)
        output = hierarchical_moe(test_input)
        print(f"✓ OptimizedHierarchicalMoEMLPBlock前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"✗ OptimizedHierarchicalMoEMLPBlock测试失败: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("1.5B权重加载验证测试")
    print("=" * 60)
    
    # 测试特定模块
    test_specific_moe_blocks()
    
    print("\n" + "=" * 60)
    print("完整模型测试")
    print("=" * 60)
    
    # 测试完整模型
    success = test_weight_loading()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！1.5B权重可以成功加载到新的高效MoE架构")
    else:
        print("✗ 测试失败，需要检查权重加载逻辑")
    print("=" * 60)