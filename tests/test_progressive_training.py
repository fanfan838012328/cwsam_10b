"""
渐进式训练支持测试

该模块测试渐进式训练功能，包括模型深度渐进式增加、层冻结和解冻机制、
从检查点恢复训练以及训练状态迁移功能。
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import shutil
import tempfile

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.progressive_training import ProgressiveTrainingManager, ProgressiveTrainingConfig
from models.config import CWSAMScalingConfig, ModelScalingConfig


class SimpleModel(nn.Module):
    """用于测试的简单模型"""
    
    def __init__(self, depth=2, embed_dim=64, num_experts=4):
        super().__init__()
        self.depth = depth
        self.embed_dim = embed_dim
        self.num_experts = num_experts
        
        # 模拟patch_embed
        self.patch_embed = nn.Linear(32, embed_dim)
        
        # 模拟位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
        
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
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        """前向传播"""
        x = self.patch_embed(x)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = x + block(x)
        
        return self.head(x.mean(dim=1))
    
    def update_structure(self, depth=None, embed_dim=None, num_experts=None):
        """更新模型结构"""
        if depth is not None and depth != self.depth:
            # 增加或减少层数
            if depth > self.depth:
                # 添加新层
                for _ in range(depth - self.depth):
                    self.blocks.append(self._make_block(self.embed_dim))
            else:
                # 移除层
                self.blocks = self.blocks[:depth]
            self.depth = depth
        
        if embed_dim is not None and embed_dim != self.embed_dim:
            # 更新嵌入维度
            old_embed_dim = self.embed_dim
            self.embed_dim = embed_dim
            
            # 更新patch_embed
            new_patch_embed = nn.Linear(32, embed_dim)
            with torch.no_grad():
                if embed_dim > old_embed_dim:
                    new_patch_embed.weight[:old_embed_dim, :] = self.patch_embed.weight
                else:
                    new_patch_embed.weight = nn.Parameter(self.patch_embed.weight[:embed_dim, :])
            self.patch_embed = new_patch_embed
            
            # 更新pos_embed
            new_pos_embed = nn.Parameter(torch.zeros(1, 16, embed_dim))
            with torch.no_grad():
                if embed_dim > old_embed_dim:
                    new_pos_embed[:, :, :old_embed_dim] = self.pos_embed
                else:
                    new_pos_embed = nn.Parameter(self.pos_embed[:, :, :embed_dim])
            self.pos_embed = new_pos_embed
            
            # 更新blocks
            new_blocks = nn.ModuleList()
            for i, block in enumerate(self.blocks):
                new_block = self._make_block(embed_dim)
                new_blocks.append(new_block)
            self.blocks = new_blocks
            
            # 更新head
            new_head = nn.Linear(embed_dim, 10)
            with torch.no_grad():
                if embed_dim > old_embed_dim:
                    new_head.weight[:, :old_embed_dim] = self.head.weight
                else:
                    new_head.weight = nn.Parameter(self.head.weight[:, :embed_dim])
            self.head = new_head


class TestProgressiveTraining(unittest.TestCase):
    """渐进式训练测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模型
        self.model = SimpleModel(depth=2, embed_dim=64, num_experts=4)
        
        # 创建优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # 创建训练配置
        self.config = ProgressiveTrainingConfig(
            initial_model_size="1.5B",
            target_model_size="3B",
            growth_strategy="depth_first",
            checkpoint_dir=os.path.join(self.temp_dir, "checkpoints")
        )
        
        # 创建训练管理器
        self.trainer = ProgressiveTrainingManager(
            model=self.model,
            config=self.config,
            optimizer=self.optimizer,
            verbose=False
        )
        
        # 创建模拟数据
        self.train_data = TensorDataset(
            torch.randn(100, 16, 32),  # 输入
            torch.randint(0, 10, (100,))  # 标签
        )
        self.train_loader = DataLoader(self.train_data, batch_size=16)
    
    def tearDown(self):
        """测试后清理"""
        # 删除临时目录
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.trainer.current_stage, 0)
        self.assertEqual(self.trainer.current_epoch, 0)
        self.assertEqual(len(self.trainer.training_history), 0)
        self.assertEqual(len(self.trainer.frozen_parameters), 0)
    
    def test_model_growth(self):
        """测试模型增长"""
        # 创建阶段配置
        stage_config = {
            "depth": 4,  # 增加深度
            "embed_dim": 64,
            "moe_num_experts": 4,
            "moe_start_layer": 2  # 确保MoE开始层小于深度
        }
        
        # 修改模型的update_structure方法以适应测试
        def mock_update_structure(target_config):
            self.model.update_structure(
                depth=target_config.depth,
                embed_dim=target_config.embed_dim,
                num_experts=target_config.moe_num_experts
            )
        
        # 替换方法
        self.trainer._update_model_structure = mock_update_structure
        
        # 模拟权重迁移
        def mock_migrate_weights(source_state_dict):
            return source_state_dict.copy()
        
        # 创建模拟的权重迁移管理器
        class MockMigrationManager:
            def __init__(self, **kwargs):
                pass
                
            def migrate_weights(self, source_state_dict):
                return source_state_dict.copy()
        
        # 保存原始的WeightMigrationManager
        original_manager = self.trainer.__class__.__module__ + ".WeightMigrationManager"
        
        # 替换为模拟版本
        setattr(sys.modules[self.trainer.__class__.__module__], "WeightMigrationManager", MockMigrationManager)
        
        try:
            # 执行模型增长
            self.trainer._grow_model(stage_config)
        finally:
            # 恢复原始版本
            from models.weight_migration import WeightMigrationManager as OriginalManager
            setattr(sys.modules[self.trainer.__class__.__module__], "WeightMigrationManager", OriginalManager)
        
        # 验证模型结构已更新
        self.assertEqual(self.model.depth, 4)
        self.assertEqual(len(self.model.blocks), 4)
    
    def test_layer_freezing(self):
        """测试层冻结"""
        # 测试渐进式解冻
        self.trainer.config.freeze_strategy = "gradual_unfreeze"
        self.trainer._setup_layer_freezing(0)
        
        # 验证部分参数被冻结
        frozen_count = len(self.trainer.frozen_parameters)
        self.assertGreater(frozen_count, 0)
        
        # 测试层级冻结
        self.trainer.config.freeze_strategy = "layer_wise"
        self.trainer._setup_layer_freezing(0)
        
        # 验证部分参数被冻结
        frozen_count = len(self.trainer.frozen_parameters)
        self.assertGreater(frozen_count, 0)
        
        # 测试解冻所有参数
        self.trainer._unfreeze_all_parameters()
        self.assertEqual(len(self.trainer.frozen_parameters), 0)
    
    def test_checkpoint_save_load(self):
        """测试检查点保存和加载"""
        # 保存检查点
        os.makedirs(os.path.join(self.temp_dir, "checkpoints"), exist_ok=True)
        checkpoint_path = os.path.join(self.temp_dir, "checkpoints", "stage_0_checkpoint.pth")
        self.trainer._save_checkpoint(0, {"test": "data"})
        
        # 修改当前状态
        self.trainer.current_stage = 99
        self.trainer.current_epoch = 99
        
        # 加载检查点
        success = self.trainer.load_checkpoint(checkpoint_path)
        
        # 验证加载成功
        self.assertTrue(success)
        self.assertEqual(self.trainer.current_stage, 0)
    
    def test_train_stage(self):
        """测试训练阶段"""
        # 创建阶段配置
        stage_config = {
            "depth": 2,
            "embed_dim": 64,
            "moe_num_experts": 4,
            "epochs": 2
        }
        
        # 执行训练
        results = self.trainer._train_stage(
            stage_config,
            self.train_loader,
            None,  # 无验证数据
            None,  # 使用默认训练函数
            None   # 无评估函数
        )
        
        # 验证结果
        self.assertEqual(len(results["train_losses"]), 2)
        self.assertEqual(self.trainer.current_epoch, 1)
    
    def test_progressive_training(self):
        """测试完整的渐进式训练流程"""
        # 修改模型的update_structure方法以适应测试
        def mock_update_structure(target_config):
            self.model.update_structure(
                depth=target_config.depth,
                embed_dim=target_config.embed_dim,
                num_experts=target_config.moe_num_experts
            )
        
        # 替换方法
        self.trainer._update_model_structure = mock_update_structure
        
        # 创建简短的增长计划
        self.trainer.config.growth_schedule = [
            {
                "stage": 1,
                "depth": 2,
                "embed_dim": 64,
                "moe_num_experts": 4,
                "moe_start_layer": 1,
                "epochs": 1
            },
            {
                "stage": 2,
                "depth": 3,
                "embed_dim": 64,
                "moe_num_experts": 4,
                "moe_start_layer": 2,
                "epochs": 1
            }
        ]
        
        # 创建模拟的权重迁移管理器
        class MockMigrationManager:
            def __init__(self, **kwargs):
                pass
                
            def migrate_weights(self, source_state_dict):
                return source_state_dict.copy()
        
        # 保存原始的WeightMigrationManager
        original_manager = self.trainer.__class__.__module__ + ".WeightMigrationManager"
        
        # 替换为模拟版本
        setattr(sys.modules[self.trainer.__class__.__module__], "WeightMigrationManager", MockMigrationManager)
        
        try:
            # 执行渐进式训练
            results = self.trainer.train_progressive(
                self.train_loader,
                None,  # 无验证数据
                None,  # 使用默认训练函数
                None   # 无评估函数
            )
        finally:
            # 恢复原始版本
            from models.weight_migration import WeightMigrationManager as OriginalManager
            setattr(sys.modules[self.trainer.__class__.__module__], "WeightMigrationManager", OriginalManager)
        
        # 验证结果
        self.assertEqual(len(self.trainer.training_history), 2)
        self.assertEqual(self.model.depth, 3)


if __name__ == "__main__":
    unittest.main()