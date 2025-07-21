"""
渐进式训练示例

该示例展示如何使用渐进式训练功能来训练一个逐步增长的模型。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import asdict

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.progressive_training import ProgressiveTrainingManager, ProgressiveTrainingConfig
from models.config import CWSAMScalingConfig


class SimpleVisionModel(nn.Module):
    """简单的视觉模型用于演示"""
    
    def __init__(self, depth=2, embed_dim=64, num_experts=4):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # 模拟patch_embed
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)
        
        # 模拟位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, embed_dim))
        
        # 模拟transformer块
        self.blocks = nn.ModuleList([
            self._make_block(embed_dim) for _ in range(depth)
        ])
        
        # 模拟输出头
        self.head = nn.Linear(embed_dim, 10)
    
    def _make_block(self, dim):
        """创建一个简单的transformer块"""
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, x):
        """前向传播"""
        # [B, 3, 256, 256] -> [B, embed_dim, 16, 16]
        x = self.patch_embed(x)
        
        # [B, embed_dim, 16, 16] -> [B, 256, embed_dim]
        x = x.flatten(2).transpose(1, 2)
        
        # 添加位置编码
        x = x + self.pos_embed[:, :x.size(1)]
        
        # 通过transformer块
        for block in self.blocks:
            x = x + block(x)
        
        # 全局池化
        x = x.mean(dim=1)
        
        # 分类头
        return self.head(x)
    
    def update_structure(self, config):
        """更新模型结构"""
        old_depth = self.depth
        old_embed_dim = self.embed_dim
        
        self.depth = config.depth
        self.embed_dim = config.embed_dim
        self.num_experts = config.moe_num_experts
        
        # 更新深度
        if config.depth > old_depth:
            # 添加新层
            for _ in range(config.depth - old_depth):
                self.blocks.append(self._make_block(self.embed_dim))
        elif config.depth < old_depth:
            # 移除层
            self.blocks = self.blocks[:config.depth]
        
        # 更新嵌入维度
        if config.embed_dim != old_embed_dim:
            # 更新patch_embed
            new_patch_embed = nn.Conv2d(3, config.embed_dim, kernel_size=16, stride=16)
            if hasattr(self, 'patch_embed'):
                # 复制权重
                with torch.no_grad():
                    if config.embed_dim > old_embed_dim:
                        new_patch_embed.weight[:old_embed_dim] = self.patch_embed.weight
                        new_patch_embed.bias[:old_embed_dim] = self.patch_embed.bias
                    else:
                        new_patch_embed.weight = nn.Parameter(self.patch_embed.weight[:config.embed_dim])
                        new_patch_embed.bias = nn.Parameter(self.patch_embed.bias[:config.embed_dim])
            self.patch_embed = new_patch_embed
            
            # 更新位置编码
            new_pos_embed = nn.Parameter(torch.zeros(1, 64, config.embed_dim))
            if hasattr(self, 'pos_embed'):
                with torch.no_grad():
                    if config.embed_dim > old_embed_dim:
                        new_pos_embed[:, :, :old_embed_dim] = self.pos_embed
                    else:
                        new_pos_embed = nn.Parameter(self.pos_embed[:, :, :config.embed_dim])
            self.pos_embed = new_pos_embed
            
            # 更新transformer块
            new_blocks = nn.ModuleList()
            for i in range(self.depth):
                if i < len(self.blocks) and config.embed_dim == old_embed_dim:
                    # 保留现有块
                    new_blocks.append(self.blocks[i])
                else:
                    # 创建新块
                    new_blocks.append(self._make_block(config.embed_dim))
            self.blocks = new_blocks
            
            # 更新输出头
            new_head = nn.Linear(config.embed_dim, 10)
            if hasattr(self, 'head'):
                with torch.no_grad():
                    if config.embed_dim > old_embed_dim:
                        new_head.weight[:, :old_embed_dim] = self.head.weight
                        new_head.bias = self.head.bias
                    else:
                        new_head.weight = nn.Parameter(self.head.weight[:, :config.embed_dim])
                        new_head.bias = self.head.bias
            self.head = new_head
        
        print(f"模型结构已更新: 深度={self.depth}, 嵌入维度={self.embed_dim}, 专家数量={self.num_experts}")


def create_dummy_dataset(num_samples=100, image_size=256):
    """创建模拟数据集"""
    images = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(images, labels)


def main():
    """主函数"""
    print("渐进式训练示例")
    print("=" * 50)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建模型
    model = SimpleVisionModel(depth=2, embed_dim=64, num_experts=4)
    model.to(device)
    print("初始模型结构:")
    print(f"- 深度: {model.depth}")
    print(f"- 嵌入维度: {model.embed_dim}")
    print(f"- 专家数量: {model.num_experts}")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建数据集
    train_dataset = create_dummy_dataset()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    # 创建渐进式训练配置
    config = ProgressiveTrainingConfig(
        initial_model_size="1.5B",
        target_model_size="3B",
        growth_strategy="depth_first",
        freeze_strategy="gradual_unfreeze",
        checkpoint_dir="checkpoints/progressive_example"
    )
    
    # 自定义增长计划
    config.growth_schedule = [
        {
            "stage": 1,
            "depth": 2,
            "embed_dim": 64,
            "moe_num_experts": 4,
            "epochs": 2
        },
        {
            "stage": 2,
            "depth": 4,
            "embed_dim": 64,
            "moe_num_experts": 4,
            "epochs": 2
        },
        {
            "stage": 3,
            "depth": 4,
            "embed_dim": 128,
            "moe_num_experts": 8,
            "epochs": 2
        }
    ]
    
    # 创建训练管理器
    trainer = ProgressiveTrainingManager(
        model=model,
        config=config,
        optimizer=optimizer,
        device=device,
        verbose=True
    )
    
    # 自定义模型结构更新函数
    def custom_update_structure(target_config):
        model.update_structure(target_config)
    
    # 替换默认的结构更新函数
    trainer._update_model_structure = custom_update_structure
    
    # 执行渐进式训练
    results = trainer.train_progressive(train_loader)
    
    # 打印训练结果
    print("\n训练完成!")
    print(f"完成的阶段数: {results['completed_stages']}/{results['total_stages']}")
    print(f"最终模型结构:")
    print(f"- 深度: {model.depth}")
    print(f"- 嵌入维度: {model.embed_dim}")
    print(f"- 专家数量: {model.num_experts}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.checkpoint_dir, "final_model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "depth": model.depth,
            "embed_dim": model.embed_dim,
            "num_experts": model.num_experts
        }
    }, final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    main()