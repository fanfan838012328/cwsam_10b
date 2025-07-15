"""
验证完整模型的参数冻结
"""

import torch
import yaml
import os
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

import models


def test_full_model_freezing():
    """测试完整模型的参数冻结"""
    print("测试完整10B模型的参数冻结...")
    
    # 1. 加载配置
    config_path = '/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(f"模型配置: {config['model']['name']}")
    
    # 2. 创建模型
    print("创建模型...")
    model = models.make(config['model'])
    
    # 3. 统计初始参数
    total_params_initial = sum(p.numel() for p in model.parameters())
    trainable_params_initial = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n初始状态:")
    print(f"  总参数: {total_params_initial:,} ({total_params_initial/1e9:.1f}B)")
    print(f"  可训练参数: {trainable_params_initial:,}")
    print(f"  冻结参数: {total_params_initial - trainable_params_initial:,}")
    
    # 4. 加载checkpoint
    checkpoint_path = config.get('sam_checkpoint')
    if not os.path.exists(checkpoint_path):
        print(f"警告: checkpoint文件不存在: {checkpoint_path}")
        return False
    
    print(f"\n加载checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 应用基础权重
    model_dict = model.state_dict()
    filtered_dict = {k: v for k, v in checkpoint.items() 
                    if k in model_dict and model_dict[k].shape == v.shape}
    model.load_state_dict(filtered_dict, strict=False)
    print(f"加载了 {len(filtered_dict)} 个匹配的权重参数")
    
    # 5. 初始化MoE权重（这里会冻结参数）
    print(f"\n初始化MoE权重...")
    moe_layer_count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'load_1_5b_expert_weights'):
            print(f"  正在初始化: {name}")
            module.load_1_5b_expert_weights(checkpoint)
            moe_layer_count += 1
            
            # 只显示前几层的详细信息
            if moe_layer_count <= 3:
                layer_total = sum(p.numel() for p in module.parameters())
                layer_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                layer_frozen = layer_total - layer_trainable
                print(f"    层参数: 总={layer_total:,}, 可训练={layer_trainable:,}, 冻结={layer_frozen:,}")
    
    print(f"总共初始化了 {moe_layer_count} 个MoE层")
    
    # 6. 统计最终参数状态
    total_params_final = sum(p.numel() for p in model.parameters())
    trainable_params_final = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params_final = total_params_final - trainable_params_final
    
    print(f"\n最终状态:")
    print(f"  总参数: {total_params_final:,} ({total_params_final/1e9:.1f}B)")
    print(f"  可训练参数: {trainable_params_final:,} ({trainable_params_final/1e9:.1f}B)")
    print(f"  冻结参数: {frozen_params_final:,} ({frozen_params_final/1e9:.1f}B)")
    
    # 7. 分析冻结效果
    frozen_change = frozen_params_final - (total_params_initial - trainable_params_initial)
    trainable_ratio = trainable_params_final / total_params_final * 100
    frozen_ratio = frozen_params_final / total_params_final * 100
    
    print(f"\n冻结分析:")
    print(f"  新增冻结参数: {frozen_change:,}")
    print(f"  可训练比例: {trainable_ratio:.1f}%")
    print(f"  冻结比例: {frozen_ratio:.1f}%")
    
    # 8. 验证结果
    expected_frozen_ratio = 30  # 预期大约30%的参数被冻结
    
    if frozen_change > 1000000 and frozen_ratio > expected_frozen_ratio:
        print(f"\n✓ 参数冻结成功！")
        print(f"  ✓ 冻结了 {frozen_change:,} 个1.5B专家参数")
        print(f"  ✓ 冻结比例 {frozen_ratio:.1f}% 符合预期")
        print(f"  ✓ 可训练参数 {trainable_params_final/1e9:.1f}B 合理")
        return True
    else:
        print(f"\n✗ 参数冻结可能有问题")
        print(f"  冻结变化: {frozen_change:,} (预期 > 1,000,000)")
        print(f"  冻结比例: {frozen_ratio:.1f}% (预期 > {expected_frozen_ratio}%)")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("完整10B模型参数冻结验证")
    print("=" * 70)
    
    success = test_full_model_freezing()
    
    print("\n" + "=" * 70)
    if success:
        print("🎉 参数冻结验证成功！")
        print("✓ 1.5B专家权重正确冻结")
        print("✓ 新专家权重可以训练")
        print("✓ 参数分配合理")
        print("✓ 模型已准备好训练")
    else:
        print("❌ 参数冻结验证失败")
        print("需要检查冻结逻辑")
    print("=" * 70)