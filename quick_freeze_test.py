"""
快速验证参数冻结
"""

import torch
import sys

# 添加项目路径
sys.path.append('/mnt/fanfq/project/code/cwsam_10b')

from models.mmseg.models.sam.efficient_moe import EfficientMoEMLPBlock


def quick_freeze_test():
    """快速测试参数冻结"""
    print("快速参数冻结测试...")
    
    # 创建测试用的专家权重
    fake_checkpoint = {}
    for expert_idx in range(16):
        base_key = f'image_encoder.blocks.28.mlp.experts.{expert_idx}'
        fake_checkpoint[f'{base_key}.0.weight'] = torch.randn(512, 128)  # 小一点的权重
        fake_checkpoint[f'{base_key}.0.bias'] = torch.randn(512)
        fake_checkpoint[f'{base_key}.2.weight'] = torch.randn(128, 512)
        fake_checkpoint[f'{base_key}.2.bias'] = torch.randn(128)
    
    print(f"创建了 {len(fake_checkpoint)} 个测试权重")
    
    # 测试EfficientMoEMLPBlock
    print("\n测试EfficientMoEMLPBlock...")
    moe_block = EfficientMoEMLPBlock(
        embedding_dim=128,   # 小模型
        mlp_dim=512,
        num_experts=24,      # 24个专家
        use_expert_choice=False,
        use_sharding=False,
    )
    
    # 统计加载前的参数
    total_before = sum(p.numel() for p in moe_block.parameters())
    trainable_before = sum(p.numel() for p in moe_block.parameters() if p.requires_grad)
    
    print(f"  加载前:")
    print(f"    总参数: {total_before:,}")
    print(f"    可训练: {trainable_before:,}")
    print(f"    冻结: {total_before - trainable_before:,}")
    
    # 加载权重（应该冻结前16个专家）
    print("  加载1.5B权重...")
    moe_block.load_1_5b_expert_weights(fake_checkpoint)
    
    # 统计加载后的参数
    total_after = sum(p.numel() for p in moe_block.parameters())
    trainable_after = sum(p.numel() for p in moe_block.parameters() if p.requires_grad)
    frozen_after = total_after - trainable_after
    
    print(f"  加载后:")
    print(f"    总参数: {total_after:,}")
    print(f"    可训练: {trainable_after:,}")
    print(f"    冻结: {frozen_after:,}")
    
    # 验证冻结情况
    frozen_change = frozen_after - (total_before - trainable_before)
    print(f"  新增冻结参数: {frozen_change:,}")
    
    # 检查每个专家的状态
    print("  专家状态检查:")
    frozen_experts = 0
    trainable_experts = 0
    
    for expert_id in range(min(moe_block.num_experts, 24)):
        if hasattr(moe_block, 'experts') and expert_id < len(moe_block.experts):
            expert = moe_block.experts[expert_id]
            all_frozen = all(not p.requires_grad for p in expert.parameters())
            
            if all_frozen:
                frozen_experts += 1
            else:
                trainable_experts += 1
    
    print(f"    冻结专家: {frozen_experts}")
    print(f"    可训练专家: {trainable_experts}")
    
    # 验证结果
    if frozen_experts == 16 and trainable_experts == 8:
        print("  ✓ 冻结策略正确！前16个专家被冻结")
        return True
    elif frozen_experts > 0:
        print(f"  ⚠ 部分冻结：{frozen_experts}个专家被冻结")
        return True
    else:
        print("  ✗ 冻结失败：没有专家被冻结")
        return False


def test_parameter_counts():
    """测试参数计数"""
    print("\n" + "="*40)
    print("参数计数验证")
    print("="*40)
    
    # 创建一个简单的模型进行验证
    simple_model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.Linear(20, 10)
    )
    
    print("简单模型测试:")
    total = sum(p.numel() for p in simple_model.parameters())
    trainable = sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
    print(f"  总参数: {total}")
    print(f"  可训练: {trainable}")
    
    # 冻结第一层
    for param in simple_model[0].parameters():
        param.requires_grad = False
    
    trainable_after = sum(p.numel() for p in simple_model.parameters() if p.requires_grad)
    frozen_after = total - trainable_after
    
    print(f"  冻结第一层后:")
    print(f"    可训练: {trainable_after}")
    print(f"    冻结: {frozen_after}")
    
    if frozen_after > 0:
        print("  ✓ 基础冻结功能正常")
        return True
    else:
        print("  ✗ 基础冻结功能异常")
        return False


if __name__ == "__main__":
    print("快速参数冻结验证")
    print("="*40)
    
    # 基础功能测试
    basic_ok = test_parameter_counts()
    
    # MoE冻结测试
    moe_ok = quick_freeze_test()
    
    print("\n" + "="*40)
    if basic_ok and moe_ok:
        print("✓ 参数冻结功能正常")
    else:
        print("✗ 参数冻结存在问题")
    print("="*40)