"""
验证参数冻结是否正确
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

import models


def test_parameter_freezing():
    """测试参数冻结功能"""
    print("测试参数冻结功能...")
    
    # 1. 加载配置
    config_path = '/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 2. 创建模型
    print("创建模型...")
    model = models.make(config['model'])
    
    # 3. 统计初始参数状态
    total_params_before = sum(p.numel() for p in model.parameters())
    trainable_params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_before = total_params_before - trainable_params_before
    
    print(f"\n权重加载前:")
    print(f"  总参数: {total_params_before:,}")
    print(f"  可训练参数: {trainable_params_before:,}")
    print(f"  冻结参数: {frozen_params_before:,}")
    
    # 4. 加载checkpoint并初始化MoE
    checkpoint_path = config.get('sam_checkpoint')
    if os.path.exists(checkpoint_path):
        print(f"\n加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 应用基础权重
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in checkpoint.items() 
                        if k in model_dict and model_dict[k].shape == v.shape}
        model.load_state_dict(filtered_dict, strict=False)
        
        # 初始化高效MoE权重（这里会冻结参数）
        print("\n初始化高效MoE权重...")
        moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'load_1_5b_expert_weights'):
                print(f"  正在初始化: {name}")
                module.load_1_5b_expert_weights(checkpoint)
                moe_layers.append((name, module))
        
        print(f"总共初始化了 {len(moe_layers)} 个MoE层")
    else:
        print(f"警告: checkpoint文件不存在: {checkpoint_path}")
        return False
    
    # 5. 统计权重加载后的参数状态
    total_params_after = sum(p.numel() for p in model.parameters())
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_after = total_params_after - trainable_params_after
    
    print(f"\n权重加载后:")
    print(f"  总参数: {total_params_after:,}")
    print(f"  可训练参数: {trainable_params_after:,}")
    print(f"  冻结参数: {frozen_params_after:,}")
    
    # 6. 分析参数变化
    frozen_change = frozen_params_after - frozen_params_before
    trainable_change = trainable_params_after - trainable_params_before
    
    print(f"\n参数变化:")
    print(f"  新增冻结参数: {frozen_change:,}")
    print(f"  可训练参数变化: {trainable_change:,}")
    print(f"  冻结比例: {frozen_params_after/total_params_after*100:.1f}%")
    print(f"  可训练比例: {trainable_params_after/total_params_after*100:.1f}%")
    
    # 7. 验证冻结是否正确
    if frozen_change > 0:
        print(f"\n✓ 参数冻结成功！冻结了 {frozen_change:,} 个参数")
        
        # 详细分析每个MoE层的冻结情况
        print("\n详细冻结情况:")
        for layer_name, moe_module in moe_layers[:3]:  # 只显示前3层
            layer_total = 0
            layer_frozen = 0
            layer_trainable = 0
            
            for param in moe_module.parameters():
                layer_total += param.numel()
                if param.requires_grad:
                    layer_trainable += param.numel()
                else:
                    layer_frozen += param.numel()
            
            print(f"  {layer_name}:")
            print(f"    总参数: {layer_total:,}")
            print(f"    冻结: {layer_frozen:,} ({layer_frozen/layer_total*100:.1f}%)")
            print(f"    可训练: {layer_trainable:,} ({layer_trainable/layer_total*100:.1f}%)")
        
        return True
    else:
        print(f"\n✗ 参数冻结失败！没有参数被冻结")
        return False


def test_specific_expert_freezing():
    """测试特定专家的冻结"""
    print("\n" + "="*50)
    print("测试特定专家冻结功能")
    print("="*50)
    
    from models.mmseg.models.sam.efficient_moe import EfficientMoEMLPBlock
    
    # 创建测试用的专家权重
    fake_checkpoint = {}
    for expert_idx in range(16):
        base_key = f'image_encoder.blocks.28.mlp.experts.{expert_idx}'
        fake_checkpoint[f'{base_key}.0.weight'] = torch.randn(5120, 1280)
        fake_checkpoint[f'{base_key}.0.bias'] = torch.randn(5120)
        fake_checkpoint[f'{base_key}.2.weight'] = torch.randn(1280, 5120)
        fake_checkpoint[f'{base_key}.2.bias'] = torch.randn(1280)
    
    # 测试EfficientMoEMLPBlock
    print("\n测试EfficientMoEMLPBlock冻结...")
    moe_block = EfficientMoEMLPBlock(
        embedding_dim=1280,
        mlp_dim=5120,
        num_experts=32,  # 32个专家，前16个应该被冻结
        use_expert_choice=True,
        use_sharding=False,
    )
    
    # 加载前统计
    total_before = sum(p.numel() for p in moe_block.parameters())
    trainable_before = sum(p.numel() for p in moe_block.parameters() if p.requires_grad)
    
    print(f"  加载前: 总参数={total_before:,}, 可训练={trainable_before:,}")
    
    # 加载权重（会自动冻结前16个专家）
    moe_block.load_1_5b_expert_weights(fake_checkpoint)
    
    # 加载后统计
    total_after = sum(p.numel() for p in moe_block.parameters())
    trainable_after = sum(p.numel() for p in moe_block.parameters() if p.requires_grad)
    frozen_after = total_after - trainable_after
    
    print(f"  加载后: 总参数={total_after:,}, 可训练={trainable_after:,}, 冻结={frozen_after:,}")
    
    # 验证前16个专家是否被冻结
    frozen_experts = 0
    trainable_experts = 0
    
    for expert_id in range(moe_block.num_experts):
        expert = moe_block.experts[expert_id]
        expert_frozen = all(not p.requires_grad for p in expert.parameters())
        
        if expert_frozen:
            frozen_experts += 1
        else:
            trainable_experts += 1
    
    print(f"  专家状态: 冻结专家={frozen_experts}, 可训练专家={trainable_experts}")
    
    if frozen_experts == 16 and trainable_experts == 16:
        print("  ✓ 专家冻结正确！前16个专家被冻结，后16个可训练")
        return True
    else:
        print("  ✗ 专家冻结错误！")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("参数冻结验证测试")
    print("=" * 60)
    
    # 测试完整模型
    success1 = test_parameter_freezing()
    
    # 测试特定模块
    success2 = test_specific_expert_freezing()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 所有冻结测试通过！")
        print("✓ 1.5B权重正确加载并冻结")
        print("✓ 新专家参数可以正常训练")
        print("✓ 参数冻结策略工作正常")
    else:
        print("❌ 部分冻结测试失败")
    print("=" * 60)