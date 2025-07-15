"""
专门测试图像编码器中的MoE实现
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

import models


def test_image_encoder_moe():
    """专门测试图像编码器的MoE实现"""
    print("测试图像编码器MoE实现...")
    
    # 1. 加载配置
    config_path = '/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"模型配置: {config['model']['name']}")
    
    # 2. 创建模型
    print("创建模型...")
    model = models.make(config['model'])
    
    # 3. 检查图像编码器
    image_encoder = None
    if hasattr(model, 'image_encoder'):
        image_encoder = model.image_encoder
        print(f"图像编码器类型: {type(image_encoder)}")
        print(f"图像编码器模块: {type(image_encoder).__module__}")
    else:
        print("未找到图像编码器!")
        return False
    
    # 4. 检查图像编码器的blocks
    if hasattr(image_encoder, 'blocks'):
        print(f"\n图像编码器层数: {len(image_encoder.blocks)}")
        
        # 检查MoE开始层
        moe_start_layer = getattr(image_encoder, 'moe_start_layer_index', 24)
        print(f"MoE开始层: {moe_start_layer}")
        
        # 检查每一层的MLP类型
        hierarchical_moe_count = 0
        optimized_moe_count = 0
        standard_mlp_count = 0
        
        for i, block in enumerate(image_encoder.blocks):
            block_type = type(block).__name__
            mlp_type = type(block.mlp).__name__ if hasattr(block, 'mlp') else 'None'
            mlp_module = type(block.mlp).__module__ if hasattr(block, 'mlp') else 'None'
            
            # 检查是否使用分层MoE
            use_hierarchical_moe = getattr(block, 'use_hierarchical_moe', False)
            
            if i < 5 or i >= moe_start_layer:  # 显示前5层和MoE层
                print(f"  Layer {i}: Block={block_type}, MLP={mlp_type}")
                print(f"           模块={mlp_module}")
                print(f"           use_hierarchical_moe={use_hierarchical_moe}")
                
                # 检查高效MoE特征
                if hasattr(block.mlp, 'load_1_5b_expert_weights'):
                    print(f"           ✓ 包含 load_1_5b_expert_weights")
                    optimized_moe_count += 1
                elif 'HierarchicalMoE' in mlp_type:
                    print(f"           原始分层MoE")
                    hierarchical_moe_count += 1
                else:
                    print(f"           标准MLP")
                    standard_mlp_count += 1
        
        print(f"\n统计结果:")
        print(f"  优化分层MoE层: {optimized_moe_count}")
        print(f"  原始分层MoE层: {hierarchical_moe_count}")
        print(f"  标准MLP层: {standard_mlp_count}")
        
        # 5. 特别检查第24层以后的层（应该使用分层MoE）
        print(f"\n检查MoE层 (Layer {moe_start_layer} 以后):")
        for i in range(moe_start_layer, len(image_encoder.blocks)):
            block = image_encoder.blocks[i]
            if hasattr(block, 'mlp'):
                mlp_type = type(block.mlp).__name__
                mlp_module = type(block.mlp).__module__
                
                print(f"  Layer {i}: {mlp_type}")
                
                # 检查关键方法
                has_load_weights = hasattr(block.mlp, 'load_1_5b_expert_weights')
                has_freeze_experts = hasattr(block.mlp, '_freeze_original_experts')
                has_expert_choice = hasattr(block.mlp, 'expert_choice_routing')
                
                print(f"           load_1_5b_expert_weights: {'✓' if has_load_weights else '✗'}")
                print(f"           _freeze_original_experts: {'✓' if has_freeze_experts else '✗'}")
                print(f"           expert_choice_routing: {'✓' if has_expert_choice else '✗'}")
                
                if has_load_weights:
                    return True
        
        return optimized_moe_count > 0
    else:
        print("图像编码器没有blocks!")
        return False


def test_hierarchical_block_creation():
    """测试HierarchicalBlock的创建过程"""
    print("\n" + "="*60)
    print("测试HierarchicalBlock创建过程")
    print("="*60)
    
    # 直接测试HierarchicalBlock的创建
    from models.mmseg.models.sam.image_encoder_moe_layer import HierarchicalBlock
    
    print("创建HierarchicalBlock...")
    try:
        block = HierarchicalBlock(
            dim=1280,
            num_heads=16,
            mlp_ratio=4.0,
            num_expert_groups=6,
            experts_per_group=16,
            k_groups=2,
            k_experts=4,
            use_hierarchical_moe=True,  # 关键参数
        )
        
        print(f"✓ HierarchicalBlock创建成功")
        print(f"  MLP类型: {type(block.mlp)}")
        print(f"  MLP模块: {type(block.mlp).__module__}")
        
        # 检查是否有高效MoE的方法
        has_load_weights = hasattr(block.mlp, 'load_1_5b_expert_weights')
        has_freeze_experts = hasattr(block.mlp, '_freeze_original_experts')
        
        print(f"  load_1_5b_expert_weights: {'✓' if has_load_weights else '✗'}")
        print(f"  _freeze_original_experts: {'✓' if has_freeze_experts else '✗'}")
        
        if has_load_weights:
            print("  ✓ 使用了OptimizedHierarchicalMoEMLPBlock")
            return True
        else:
            print("  ✗ 使用了原始HierarchicalMoEMLPBlock")
            return False
            
    except Exception as e:
        print(f"✗ HierarchicalBlock创建失败: {e}")
        return False


def test_import_in_context():
    """在HierarchicalBlock的上下文中测试导入"""
    print("\n测试在HierarchicalBlock上下文中的导入:")
    
    # 模拟HierarchicalBlock.__init__中的导入逻辑
    use_hierarchical_moe = True
    
    if use_hierarchical_moe:
        print("  尝试导入OptimizedHierarchicalMoEMLPBlock...")
        try:
            from models.mmseg.models.sam.efficient_moe import OptimizedHierarchicalMoEMLPBlock
            print("  ✓ 导入成功")
            
            # 尝试创建实例
            try:
                mlp = OptimizedHierarchicalMoEMLPBlock(
                    embedding_dim=1280,
                    mlp_dim=5120,
                    num_expert_groups=6,
                    experts_per_group=16,
                    k_groups=2,
                    k_experts=4,
                    expert_capacity_factor=1.5,
                    use_checkpoint=False,
                )
                print("  ✓ 实例化成功")
                print("  ✓ 应该使用OptimizedHierarchicalMoEMLPBlock")
                return True
            except Exception as e:
                print(f"  ✗ 实例化失败: {e}")
                print("  将回退到HierarchicalMoEMLPBlock")
                return False
                
        except ImportError as e:
            print(f"  ✗ 导入失败: {e}")
            print("  将回退到HierarchicalMoEMLPBlock")
            return False


if __name__ == "__main__":
    print("=" * 70)
    print("图像编码器MoE实现专项测试")
    print("=" * 70)
    
    # 测试导入
    import_ok = test_import_in_context()
    
    # 测试HierarchicalBlock创建
    block_ok = test_hierarchical_block_creation()
    
    # 测试完整模型的图像编码器
    model_ok = test_image_encoder_moe()
    
    print("\n" + "=" * 70)
    print("最终结果:")
    print("=" * 70)
    
    if import_ok and block_ok and model_ok:
        print("🎉 图像编码器正确使用了高效MoE实现！")
        print("✓ 导入成功")
        print("✓ HierarchicalBlock创建正确")
        print("✓ 模型中使用了OptimizedHierarchicalMoEMLPBlock")
    elif import_ok and block_ok and not model_ok:
        print("⚠ 导入和创建都正常，但模型中未使用：")
        print("✓ 导入成功")
        print("✓ HierarchicalBlock创建正确")
        print("✗ 模型可能有配置问题")
    elif import_ok and not block_ok:
        print("⚠ 导入成功但创建失败：")
        print("✓ 导入成功")
        print("✗ HierarchicalBlock创建失败")
        print("✗ 可能有参数不兼容问题")
    else:
        print("❌ 存在根本性问题：")
        print(f"  导入: {'✓' if import_ok else '✗'}")
        print(f"  创建: {'✓' if block_ok else '✗'}")
        print(f"  模型: {'✓' if model_ok else '✗'}")
    
    print("=" * 70)