"""
渐进式训练支持实现

该模块实现了渐进式训练功能，支持模型深度渐进式增加、层冻结和解冻机制、
从检查点恢复训练以及训练状态迁移功能。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import os
import json
import logging
import warnings
import math
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from collections import OrderedDict
import copy
from dataclasses import dataclass
import re
from dataclasses import asdict

try:
    from .config import CWSAMScalingConfig, ModelScalingConfig
    from .weight_migration import WeightMigrationManager
except ImportError:
    from config import CWSAMScalingConfig, ModelScalingConfig
    from weight_migration import WeightMigrationManager


@dataclass
class ProgressiveTrainingConfig:
    """渐进式训练配置数据类"""
    
    # 基础配置
    initial_model_size: str = "1.5B"
    target_model_size: str = "3B"
    
    # 渐进式增长配置
    growth_strategy: str = "depth_first"  # "depth_first", "width_first", "mixed"
    growth_schedule: List[Dict[str, Any]] = None  # 自定义增长计划
    
    # 层冻结配置
    freeze_strategy: str = "gradual_unfreeze"  # "gradual_unfreeze", "layer_wise", "block_wise", "none"
    freeze_epochs_per_stage: int = 5  # 每次解冻的层数比例
    unfreeze_rate: float = 0.2  # 每个阶段的学习率衰减
    
    # 训练配置
    warmup_epochs_per_stage: int = 2
    learning_rate_decay: float = 0.9  # 每个阶段的学习率衰减
    gradient_accumulation_steps: int = 1
    
    # 检查点配置
    save_checkpoint_every_stage: bool = True
    keep_intermediate_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints/progressive"
    
    # 验证配置
    validate_every_stage: bool = True
    early_stopping_patience: int = 3
    
    def __post_init__(self):
        """初始化后处理"""
        if self.growth_schedule is None:
            self.growth_schedule = self._create_default_growth_schedule()
    
    def _create_default_growth_schedule(self) -> List[Dict[str, Any]]:
        """创建默认的增长计划"""
        # 根据初始和目标模型大小创建增长计划
        initial_config = CWSAMScalingConfig(self.initial_model_size).get_config()
        target_config = CWSAMScalingConfig(self.target_model_size).get_config()
        
        schedule = []
        
        if self.growth_strategy == "depth_first":
            # 优先增加深度
            current_depth = initial_config.depth
            target_depth = target_config.depth
            
            # 先增加深度
            while current_depth < target_depth:
                depth_increment = max(1, (target_depth - current_depth) // 3)
                next_depth = min(current_depth + depth_increment, target_depth)
                
                schedule.append({
                    "stage": len(schedule) + 1,
                    "depth": next_depth,
                    "embed_dim": initial_config.embed_dim,
                    "moe_num_experts": initial_config.moe_num_experts,
                    "epochs": 10
                })
                
                current_depth = next_depth
            
            # 然后增加其他维度
            if target_config.embed_dim > initial_config.embed_dim:
                schedule.append({
                    "stage": len(schedule) + 1,
                    "depth": target_depth,
                    "embed_dim": target_config.embed_dim,
                    "moe_num_experts": initial_config.moe_num_experts,
                    "epochs": 15
                })
            
            if target_config.moe_num_experts > initial_config.moe_num_experts:
                schedule.append({
                    "stage": len(schedule) + 1,
                    "depth": target_depth,
                    "embed_dim": target_config.embed_dim,
                    "moe_num_experts": target_config.moe_num_experts,
                    "epochs": 20
                })
                
        elif self.growth_strategy == "width_first":
            # 优先增加宽度（嵌入维度和专家数量）
            if target_config.embed_dim > initial_config.embed_dim:
                schedule.append({
                    "stage": 1,
                    "depth": initial_config.depth,
                    "embed_dim": target_config.embed_dim,
                    "moe_num_experts": initial_config.moe_num_experts,
                    "epochs": 15
                })
            
            if target_config.moe_num_experts > initial_config.moe_num_experts:
                schedule.append({
                    "stage": len(schedule) + 1,
                    "depth": initial_config.depth,
                    "embed_dim": target_config.embed_dim,
                    "moe_num_experts": target_config.moe_num_experts,
                    "epochs": 20
                })
            
            # 最后增加深度
            if target_config.depth > initial_config.depth:
                schedule.append({
                    "stage": len(schedule) + 1,
                    "depth": target_config.depth,
                    "embed_dim": target_config.embed_dim,
                    "moe_num_experts": target_config.moe_num_experts,
                    "epochs": 10
                })
                
        else:  # mixed
            # 混合策略，同时增加各个维度
            schedule.append({
                "stage": 1,
                "depth": initial_config.depth,
                "embed_dim": initial_config.embed_dim,
                "moe_num_experts": initial_config.moe_num_experts,
                "epochs": 10
            })
            
            schedule.append({
                "stage": 2,
                "depth": (initial_config.depth + target_config.depth) // 2,
                "embed_dim": (initial_config.embed_dim + target_config.embed_dim) // 2,
                "moe_num_experts": min(initial_config.moe_num_experts * 2, target_config.moe_num_experts),
                "epochs": 15
            })
            
            schedule.append({
                "stage": 3,
                "depth": target_config.depth,
                "embed_dim": target_config.embed_dim,
                "moe_num_experts": target_config.moe_num_experts,
                "epochs": 20
            })
        
        return schedule


class ProgressiveTrainingManager:
    """
    渐进式训练管理器
    
    支持功能：
    - 模型深度渐进式增加
    - 层冻结和解冻机制
    - 从检查点恢复训练
    - 训练状态迁移功能
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: ProgressiveTrainingConfig,
        optimizer: optim.Optimizer = None,
        scheduler: _LRScheduler = None,
        device: torch.device = None,
        verbose: bool = True
    ):
        """
        初始化渐进式训练管理器
        
        Args:
            model: 要训练的模型
            config: 渐进式训练配置
            optimizer: 优化器
            scheduler: 学习率调度器
            device: 训练设备
            verbose: 是否输出详细信息
        """
        # 训练状态
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        
        # 训练状态
        self.current_stage = 0
        self.current_epoch = 0
        self.training_history = []
        self.frozen_parameters = set()
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 创建检查点目录
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # 初始化训练状态
        self._initialize_training_state()
        
        if self.verbose:
            self._print_training_plan()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"ProgressiveTraining_{id(self)}")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_training_state(self):
        """初始化训练状态"""
        self.training_state = {
            "current_stage": 0,
            "current_epoch": 0,
            "completed_stages": [],
            "training_history": [],
            "best_metrics": {},
            "early_stopping_counter": 0,
            "model_configs": []
        }
        
        # 记录初始模型配置
        initial_config = CWSAMScalingConfig(self.config.initial_model_size).get_config()
        self.training_state["model_configs"].append({
            "config": asdict(initial_config),
            "stage": 0
        })
    
    def _print_training_plan(self):
        """打印训练计划"""
        print(f"\n===== 渐进式训练计划 =====")
        print(f"初始模型: {self.config.initial_model_size}")
        print(f"目标模型: {self.config.target_model_size}")
        print(f"增长策略: {self.config.growth_strategy}")
        print(f"冻结策略: {self.config.freeze_strategy}")
        print(f"\n训练阶段:")
        
        for i, stage in enumerate(self.config.growth_schedule):
            print(f"  阶段 {i+1}: {stage}")
        
        print("=" * 30)
    
    def train_progressive(
        self,
        train_dataloader,
        val_dataloader=None,
        train_fn: Callable = None,
        eval_fn: Callable = None
    ) -> Dict[str, Any]:
        """
        执行渐进式训练
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            train_fn: 自定义训练函数
            eval_fn: 自定义评估函数
            
        Returns:
            训练结果字典
        """
        self.logger.info("开始渐进式训练...")
        
        total_stages = len(self.config.growth_schedule)
        
        for stage_idx, stage_config in enumerate(self.config.growth_schedule):
            self.current_stage = stage_idx
            
            self.logger.info(f"开始训练阶段 {stage_idx + 1}/{total_stages}")
            
            # 1. 模型增长
            if stage_idx > 0:
                self._grow_model(stage_config)
            
            # 2. 设置层冻结
            self._setup_layer_freezing(stage_idx)
            
            # 3. 调整优化器和调度器
            self._adjust_optimizer_and_scheduler(stage_idx)
            
            # 4. 训练当前阶段
            stage_results = self._train_stage(
                stage_config,
                train_dataloader,
                val_dataloader,
                train_fn,
                eval_fn
            )
            
            # 5. 保存检查点
            if self.config.save_checkpoint_every_stage:
                self._save_checkpoint(stage_idx, stage_results)
                
            # 6. 更新训练历史
            self.training_history.append({
                "stage": stage_idx,
                "config": stage_config,
                "results": stage_results
            })
            
            # 7. 早停检查
            if self._should_early_stop(stage_results):
                self.logger.info(f"早停触发，在阶段 {stage_idx + 1} 停止训练")
                break
        
        self.logger.info("渐进式训练完成")
        return self._get_training_summary()
    
    def _grow_model(self, stage_config: Dict[str, Any]):
        """
        增长模型
        """
        self.logger.info("执行模型增长...")
        
        # 获取当前和目标配置
        current_config = self._get_current_model_config()
        target_config = self._create_stage_config(stage_config)
        
        # 检查是否需要增长
        if self._configs_equal(current_config, target_config):
            self.logger.info("模型配置无变化，跳过增长")
            return
        
        # 创建权重迁移管理器
        migration_manager = WeightMigrationManager(
            source_config=current_config,
            target_config=target_config,
            verbose=self.verbose
        )
        
        # 获取当前模型权重
        current_state_dict = self.model.state_dict()
        
        # 执行权重迁移
        new_state_dict = migration_manager.migrate_weights(current_state_dict)
        
        # 更新模型结构（这里需要根据具体模型实现）
        self._update_model_structure(target_config)
        
        # 加载新权重
        self.model.load_state_dict(new_state_dict, strict=False)
        
        # 更新训练状态
        self.training_state["model_configs"].append({
            "config": asdict(target_config),
            "stage": self.current_stage
        })
        
        self.logger.info("模型增长完成")
    
    def _setup_layer_freezing(self, stage_idx: int):
        """
        设置层冻结
        """
        self.logger.info(f"设置阶段 {stage_idx + 1} 的层冻结...")
        
        # 清除之前的冻结状态
        self._unfreeze_all_parameters()
        
        if self.config.freeze_strategy == "none":
            return
        
        elif self.config.freeze_strategy == "gradual_unfreeze":
            self._gradual_unfreeze_strategy(stage_idx)
            
        elif self.config.freeze_strategy == "layer_wise":
            self._layer_wise_freeze_strategy(stage_idx)
            
        elif self.config.freeze_strategy == "block_wise":
            self._block_wise_freeze_strategy(stage_idx)
        
        frozen_count = len(self.frozen_parameters)
        total_params = sum(1 for _ in self.model.parameters())
        
        self.logger.info(f"冻结了 {frozen_count}/{total_params} 个参数")
    
    def _gradual_unfreeze_strategy(self, stage_idx: int):
        """
        渐进式解冻策略
        """
        # 在早期阶段冻结更多层，随着训练进行逐渐解冻
        total_stages = len(self.config.growth_schedule)
        unfreeze_ratio = (stage_idx + 1) / total_stages
        
        # 获取所有可训练参数
        all_params = list(self.model.named_parameters())
        
        # 计算需要解冻的参数数量
        num_to_unfreeze = int(len(all_params) * unfreeze_ratio)
        
        # 从后向前解冻（通常后面的层更重要）
        for i, (name, param) in enumerate(reversed(all_params)):
            if i < num_to_unfreeze:
                param.requires_grad = True
            else:
                param.requires_grad = False
                self.frozen_parameters.add(name)
    
    def _layer_wise_freeze_strategy(self, stage_idx: int):
        """
        按层冻结策略
        """
        # 冻结前面的层，解冻后面的层
        for name, param in self.model.named_parameters():
            if "blocks." in name:
                # 提取层索引
                layer_idx = self._extract_layer_index(name)
                
                # 根据当前阶段决定是否冻结
                freeze_threshold = max(0, 32 - (stage_idx + 1) * 8)
                
                if layer_idx < freeze_threshold:
                    param.requires_grad = False
                    self.frozen_parameters.add(name)
                else:
                    param.requires_grad = True
            else:
                # 非transformer块的参数保持可训练
                param.requires_grad = True
    
    def _block_wise_freeze_strategy(self, stage_idx: int):
        """
        按块冻结策略
        """
        # 按功能块冻结（如patch_embed, pos_embed等）
        if stage_idx == 0:
            freeze_blocks = ["patch_embed", "pos_embed"]
        elif stage_idx == 1:
            freeze_blocks = []
        
        # 后续阶段不冻结任何块
        
        for name, param in self.model.named_parameters():
            should_freeze = any(block in name for block in freeze_blocks)
            
            if should_freeze:
                param.requires_grad = False
                self.frozen_parameters.add(name)
            else:
                param.requires_grad = True
    
    def _unfreeze_all_parameters(self):
        """
        解冻所有参数
        """
        self.frozen_parameters.clear()
        
        for param in self.model.parameters():
            param.requires_grad = True
    
    def _adjust_optimizer_and_scheduler(self, stage_idx: int):
        """
        调整优化器和调度器
        """
        if self.optimizer is None:
            return
        
        # 调整学习率
        if stage_idx > 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            new_lr = current_lr * self.config.learning_rate_decay
            
            self.logger.info(f"调整学习率: {current_lr:.6f} -> {new_lr:.6f}")
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # 重新创建优化器以包含新参数
        if stage_idx > 0:
            self._recreate_optimizer()
    
    def _recreate_optimizer(self):
        """
        重新创建优化器以包含新参数
        """
        if self.optimizer is None:
            return
        
        # 获取当前优化器配置
        optimizer_class = type(self.optimizer)
        optimizer_state = self.optimizer.state_dict()
        
        # 获取可训练参数
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # 重新创建优化器
        if optimizer_class == optim.Adam:
            self.optimizer = optim.Adam(
                trainable_params,
                lr=self.optimizer.param_groups[0]['lr'],
                betas=self.optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                eps=self.optimizer.param_groups[0].get('eps', 1e-8),
                weight_decay=self.optimizer.param_groups[0].get('weight_decay', 0)
            )
        elif optimizer_class == optim.AdamW:
            self.optimizer = optim.AdamW(
                trainable_params,
                lr=self.optimizer.param_groups[0]['lr'],
                betas=self.optimizer.param_groups[0].get('betas', (0.9, 0.999)),
                eps=self.optimizer.param_groups[0].get('eps', 1e-8),
                weight_decay=self.optimizer.param_groups[0].get('weight_decay', 0.01)
            )
        
        # 尝试恢复优化器状态（对于新参数可能会失败）
        try:
            self.optimizer.load_state_dict(optimizer_state)
        except:
            self.logger.warning("无法完全恢复优化器状态，使用默认初始化")
    
    def _train_stage(
        self,
        stage_config: Dict[str, Any],
        train_dataloader,
        val_dataloader,
        train_fn: Callable,
        eval_fn: Callable
    ) -> Dict[str, Any]:
        """
        训练单个阶段
        """
        stage_epochs = stage_config.get("epochs", 10)
        
        stage_results = {
            "train_losses": [],
            "val_metrics": [],
            "best_metric": float('inf'),
            "best_epoch": 0
        }
        
        for epoch in range(stage_epochs):
            self.current_epoch = epoch
            
            # 训练
            if train_fn:
                train_loss = train_fn(
                    self.model, train_dataloader, self.optimizer, self.device, epoch
                )
            else:
                train_loss = self._default_train_epoch(
                    train_dataloader, epoch
                )
            
            stage_results["train_losses"].append(train_loss)
            
            # 验证
            if val_dataloader and eval_fn:
                val_metrics = eval_fn(
                    self.model, val_dataloader, self.device, epoch
                )
                
                stage_results["val_metrics"].append(val_metrics)
                
                # 更新最佳指标
                current_metric = val_metrics.get("loss", float('inf'))
                if current_metric < stage_results["best_metric"]:
                    stage_results["best_metric"] = current_metric
                    stage_results["best_epoch"] = epoch
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            # 日志输出
            if self.verbose and epoch % 5 == 0:
                self.logger.info(
                    f"阶段 {self.current_stage + 1}, "
                    f"Epoch {epoch + 1}/{stage_epochs}, "
                    f"Train Loss: {train_loss:.4f}"
                )
        
        return stage_results
    
    def _default_train_epoch(self, train_dataloader, epoch: int) -> float:
        """
        默认的训练epoch函数
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 这里需要根据具体的数据格式和模型接口来实现
            # 简化实现，实际使用时需要替换
            
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                inputs, targets = batch[0], batch[1]
            else:
                # 假设batch是字典格式
                inputs = batch.get("image", batch.get("input"))
                targets = batch.get("mask", batch.get("target"))
                
            if inputs is None or targets is None:
                continue
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失（简化实现）
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                # 假设是分割任务
                loss = nn.functional.cross_entropy(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 梯度累积
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def _should_early_stop(self, stage_results: Dict[str, Any]) -> bool:
        """
        检查是否应该早停
        """
        if not self.config.validate_every_stage:
            return False
        
        if not stage_results.get("val_metrics"):
            return False
        
        # 简化的早停逻辑
        current_metric = stage_results["val_metrics"][-1].get("loss", float('inf'))
        
        if current_metric < self.training_state["best_metrics"].get("loss", float('inf')):
            self.training_state["best_metrics"]["loss"] = current_metric
            self.training_state["early_stopping_counter"] = 0
        else:
            self.training_state["early_stopping_counter"] += 1
        
        return self.training_state["early_stopping_counter"] >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, stage_idx: int, stage_results: Dict[str, Any]):
        """
        保存检查点
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"stage_{stage_idx}_checkpoint.pth"
        )
        
        checkpoint = {
            "stage": stage_idx,
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "training_state": self.training_state,
            "stage_results": stage_results,
            "config": asdict(self.config)
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"检查点已保存: {checkpoint_path}")
        
        # 保存最佳模型
        if stage_results.get("best_epoch") is not None:
            best_checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                f"stage_{stage_idx}_best.pth"
            )
            torch.save(checkpoint, best_checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载检查点
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # 恢复模型状态
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # 恢复优化器状态
            if self.optimizer and checkpoint.get("optimizer_state_dict"):
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # 恢复调度器状态
            if self.scheduler and checkpoint.get("scheduler_state_dict"):
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # 恢复训练状态
            self.training_state = checkpoint.get("training_state", {})
            
            self.current_stage = checkpoint.get("stage", 0)
            self.current_epoch = checkpoint.get("epoch", 0)
            
            self.logger.info(f"检查点加载成功: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"检查点加载失败: {e}")
            return False
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """
        获取训练摘要
        """
        return {
            "config": asdict(self.config),
            "training_state": self.training_state,
            "training_history": self.training_history,
            "completed_stages": len(self.training_history),
            "total_stages": len(self.config.growth_schedule)
        }
    
    def _get_current_model_config(self) -> ModelScalingConfig:
        """
        获取当前模型配置
        """
        if self.training_state["model_configs"]:
            config_dict = self.training_state["model_configs"][-1]["config"]
            return ModelScalingConfig(**config_dict)
        else:
            return CWSAMScalingConfig(self.config.initial_model_size).get_config()
    
    def _create_stage_config(self, stage_config: Dict[str, Any]) -> ModelScalingConfig:
        """
        根据阶段配置创建模型配置
        """
        current_config = self._get_current_model_config()
        
        # 更新模型配置
        return ModelScalingConfig(
            model_size=f"stage_{self.current_stage}",
            depth=stage_config.get("depth", current_config.depth),
            embed_dim=stage_config.get("embed_dim", current_config.embed_dim),
            moe_num_experts=stage_config.get("moe_num_experts", current_config.moe_num_experts),
            num_heads=stage_config.get("num_heads", current_config.num_heads),
            moe_start_layer=stage_config.get("moe_start_layer", current_config.moe_start_layer),
            use_multi_scale=stage_config.get("use_multi_scale", current_config.use_multi_scale),
            decoder_depth=stage_config.get("decoder_depth", current_config.decoder_depth)
        )
    
    def _configs_equal(self, config1: ModelScalingConfig, config2: ModelScalingConfig) -> bool:
        """
        比较两个配置是否相等
        """
        return (
            config1.depth == config2.depth and
            config1.embed_dim == config2.embed_dim and
            config1.moe_num_experts == config2.moe_num_experts and
            config1.num_heads == config2.num_heads and
            config1.moe_start_layer == config2.moe_start_layer and
            config1.use_multi_scale == config2.use_multi_scale and
            config1.decoder_depth == config2.decoder_depth
        )
    
    def _update_model_structure(self, target_config: ModelScalingConfig):
        """
        更新模型结构（需要根据具体模型实现）
        """
        # 这里需要具体的模型类来实现结构更新
        # 简化实现，实际使用时需要替换
        pass
    
    def _extract_layer_index(self, param_name: str) -> int:
        """
        从参数名中提取层索引
        """
        import re
        match = re.search(r'blocks\.(\d+)\.', param_name)
        return int(match.group(1)) if match else 0


def create_progressive_training_manager(
    model: nn.Module,
    initial_model_size: str,
    target_model_size: str,
    optimizer: optim.Optimizer = None,
    **kwargs
) -> ProgressiveTrainingManager:
    """
    创建渐进式训练管理器的便捷函数
    
    Args:
        model: 要训练的模型
        initial_model_size: 初始模型规模
        target_model_size: 目标模型规模
        optimizer: 优化器
        **kwargs: 其他配置参数
        
    Returns:
        ProgressiveTrainingManager实例
    """
    config = ProgressiveTrainingConfig(
        initial_model_size=initial_model_size,
        target_model_size=target_model_size,
        **kwargs
    )
    
    return ProgressiveTrainingManager(
        model=model,
        config=config,
        optimizer=optimizer,
        **kwargs
    )


# 测试渐进式训练管理器
if __name__ == "__main__":
    print("测试渐进式训练管理器")
    print("=" * 50)
    
    # 创建模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        
        def forward(self, x):
            return self.linear(x)
    
    model = MockModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建渐进式训练管理器
    config = ProgressiveTrainingConfig(
        initial_model_size="1.5B",
        target_model_size="3B",
        growth_strategy="depth_first"
    )
    
    trainer = ProgressiveTrainingManager(
        model=model,
        config=config,
        optimizer=optimizer,
        verbose=True
    )
    
    print("✓ 渐进式训练管理器创建成功")
    
    # 测试检查点保存和加载
    checkpoint_path = "./test_checkpoint.pth"
    trainer._save_checkpoint(0, {"testing": "data"})
    
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
        print("✓ 检查点保存和加载测试成功")
    else:
        print("✗ 检查点保存和加载测试失败")
    
    # 清理测试文件
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)