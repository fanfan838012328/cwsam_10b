"""
验证模型是否使用了高效MoE实现
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

import models


def test_moe_implementation():
    """测试模型实际使用的MoE实现"""
    print("验证模型使用的MoE实现...")
    
    # 1. 加载配置
    config_path = '/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"模型配置: {config['model']['name']}")
    
    # 2. 创建模型
    print("创建模型...")
    model = models.make(config['model'])
    
    # 3. 检查MoE层实现
    print("\n检查MoE层实现:")
    moe_layers = []
    hierarchical_layers = []
    
    for name, module in model.named_modules():
        # 检查是否有MoE层
        if hasattr(module, 'mlp') and hasattr(module.mlp, 'experts'):
            moe_layers.append((name, module))
            
            # 检查具体的MoE类型
            mlp_type = type(module.mlp).__name__
            mlp_module = type(module.mlp).__module__
            
            print(f"  {name}:")
            print(f"    MLP类型: {mlp_type}")
            print(f"    模块路径: {mlp_module}")
            
            # 检查是否有高效MoE的特征方法
            efficient_features = []
            if hasattr(module.mlp, 'load_1_5b_expert_weights'):
                efficient_features.append('load_1_5b_expert_weights')
            if hasattr(module.mlp, '_freeze_original_experts'):
                efficient_features.append('_freeze_original_experts')
            if hasattr(module.mlp, 'expert_choice_routing'):
                efficient_features.append('expert_choice_routing')
            if hasattr(module.mlp, 'use_sharding'):
                efficient_features.append('use_sharding')
            
            if efficient_features:
                print(f"    高效MoE特征: {efficient_features}")
            else:
                print(f"    高效MoE特征: 无")
                
        # 检查HierarchicalBlock
        if 'HierarchicalBlock' in str(type(module)):
            hierarchical_layers.append((name, module))
    
    print(f"\n总共找到 {len(moe_layers)} 个MoE层")
    print(f"总共找到 {len(hierarchical_layers)} 个HierarchicalBlock层")
    
    # 4. 详细分析第一个MoE层
    if moe_layers:
        print(f"\n详细分析第一个MoE层:")
        name, module = moe_layers[0]
        print(f"  层名称: {name}")
        print(f"  MLP类型: {type(module.mlp)}")
        print(f"  MLP模块: {type(module.mlp).__module__}")
        
        # 检查专家数量
        if hasattr(module.mlp, 'num_experts'):
            print(f"  专家数量: {module.mlp.num_experts}")
        if hasattr(module.mlp, 'experts') and hasattr(module.mlp.experts, '__len__'):
            print(f"  实际专家数: {len(module.mlp.experts)}")
            
        # 检查方法存在性
        methods_to_check = [
            'load_1_5b_expert_weights',
            '_freeze_original_experts', 
            'expert_choice_routing',
            'forward'
        ]
        
        print(f"  方法检查:")
        for method in methods_to_check:
            has_method = hasattr(module.mlp, method)
            print(f"    {method}: {'✓' if has_method else '✗'}")
    
    # 5. 检查import是否成功
    print(f"\n检查模块导入状态:")
    try:
        from models.mmseg.models.sam.efficient_moe import OptimizedHierarchicalMoEMLPBlock
        print("  ✓ OptimizedHierarchicalMoEMLPBlock 导入成功")
        print(f"    模块路径: {OptimizedHierarchicalMoEMLPBlock.__module__}")
    except ImportError as e:
        print(f"  ✗ OptimizedHierarchicalMoEMLPBlock 导入失败: {e}")
    
    try:
        from models.mmseg.models.sam.common import HierarchicalMoEMLPBlock
        print("  ✓ HierarchicalMoEMLPBlock 导入成功")
        print(f"    模块路径: {HierarchicalMoEMLPBlock.__module__}")
    except ImportError as e:
        print(f"  ✗ HierarchicalMoEMLPBlock 导入失败: {e}")
    
    # 6. 判断结果
    using_efficient = False
    if moe_layers:
        first_moe = moe_layers[0][1]
        if hasattr(first_moe.mlp, 'load_1_5b_expert_weights'):
            using_efficient = True
    
    print(f"\n结果判断:")
    if using_efficient:
        print("  ✓ 模型使用了高效MoE实现")
        print("  ✓ 包含权重加载和冻结功能")
    else:
        print("  ✗ 模型使用了原始MoE实现")
        print("  ✗ 可能回退到了HierarchicalMoEMLPBlock")
    
    return using_efficient


def test_import_directly():
    """直接测试import能否成功"""
    print("\n" + "="*50)
    print("直接测试模块导入")
    print("="*50)
    
    # 测试直接导入
    print("1. 测试直接导入OptimizedHierarchicalMoEMLPBlock:")
    try:
        from models.mmseg.models.sam.efficient_moe import OptimizedHierarchicalMoEMLPBlock
        print("   ✓ 导入成功")
        
        # 尝试创建实例
        try:
            test_moe = OptimizedHierarchicalMoEMLPBlock(
                embedding_dim=128,
                mlp_dim=512,
                num_expert_groups=2,
                experts_per_group=4,
            )
            print("   ✓ 实例化成功")
            print(f"   ✓ 类型: {type(test_moe)}")
            return True
        except Exception as e:
            print(f"   ✗ 实例化失败: {e}")
            return False
            
    except ImportError as e:
        print(f"   ✗ 导入失败: {e}")
        return False


def test_hierarchical_block_import():
    """测试HierarchicalBlock中的导入"""
    print("\n2. 测试HierarchicalBlock中的import路径:")
    
    # 模拟HierarchicalBlock中的导入逻辑
    try:
        from models.mmseg.models.sam.efficient_moe import OptimizedHierarchicalMoEMLPBlock
        print("   ✓ efficient_moe.OptimizedHierarchicalMoEMLPBlock 导入成功")
        efficient_available = True
    except ImportError as e:
        print(f"   ✗ efficient_moe.OptimizedHierarchicalMoEMLPBlock 导入失败: {e}")
        efficient_available = False
    
    try:
        from models.mmseg.models.sam.common import HierarchicalMoEMLPBlock
        print("   ✓ common.HierarchicalMoEMLPBlock 导入成功")
        common_available = True
    except ImportError as e:
        print(f"   ✗ common.HierarchicalMoEMLPBlock 导入失败: {e}")
        common_available = False
    
    print(f"\n   导入结果:")
    print(f"     高效MoE可用: {efficient_available}")
    print(f"     原始MoE可用: {common_available}")
    
    if efficient_available:
        print("   ✓ 应该使用OptimizedHierarchicalMoEMLPBlock")
    elif common_available:
        print("   ⚠ 回退到HierarchicalMoEMLPBlock")
    else:
        print("   ✗ 两个实现都不可用")
    
    return efficient_available


if __name__ == "__main__":
    print("=" * 60)
    print("MoE实现验证测试")
    print("=" * 60)
    
    # 测试直接导入
    import_ok = test_import_directly()
    
    # 测试HierarchicalBlock导入路径
    hierarchical_import_ok = test_hierarchical_block_import()
    
    # 测试模型中的实际使用
    model_using_efficient = test_moe_implementation()
    
    print("\n" + "=" * 60)
    print("最终结果:")
    print("=" * 60)
    
    if import_ok and hierarchical_import_ok and model_using_efficient:
        print("🎉 所有测试通过！")
        print("✓ 高效MoE模块可以正常导入")
        print("✓ HierarchicalBlock导入路径正确")
        print("✓ 模型实际使用了高效MoE实现")
    elif import_ok and hierarchical_import_ok and not model_using_efficient:
        print("⚠ 部分问题:")
        print("✓ 高效MoE模块可以正常导入")
        print("✓ HierarchicalBlock导入路径正确")
        print("✗ 但模型可能没有使用高效MoE实现")
        print("需要检查HierarchicalBlock的创建逻辑")
    else:
        print("❌ 存在问题:")
        print(f"  模块导入: {'✓' if import_ok else '✗'}")
        print(f"  导入路径: {'✓' if hierarchical_import_ok else '✗'}")
        print(f"  模型使用: {'✓' if model_using_efficient else '✗'}")
    
    print("=" * 60)