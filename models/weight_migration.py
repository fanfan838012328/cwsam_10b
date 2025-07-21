"""
权重迁移系统实现

该模块实现了WeightMigrationManager类，支持从小模型到大模型的权重迁移，
包括MoE专家权重的复制和初始化、维度扩展时的权重插值，以及权重迁移验证机制。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import OrderedDict
import copy
import logging

try:
    from .config import CWSAMScalingConfig, ModelScalingConfig, ConfigValidator
except ImportError:
    from config import CWSAMScalingConfig, ModelScalingConfig, ConfigValidator


class WeightMigrationManager:
    """
    权重迁移管理器
    
    支持功能：
    - 从小模型到大模型的权重迁移
    - MoE专家权重的复制和初始化
    - 维度扩展时的权重插值
    - 权重迁移验证机制
    """
    
    def __init__(
        self,
        source_config: Union[CWSAMScalingConfig, ModelScalingConfig],
        target_config: Union[CWSAMScalingConfig, ModelScalingConfig],
        migration_strategy: str = "adaptive",
        expert_initialization: str = "copy_random",
        dimension_interpolation: str = "linear",
        validation_enabled: bool = True,
        verbose: bool = True
    ):
        """
        初始化权重迁移管理器
        
        Args:
            source_config: 源模型配置
            target_config: 目标模型配置
            migration_strategy: 迁移策略 ("conservative", "adaptive", "aggressive")
            expert_initialization: 专家初始化策略 ("copy_random", "copy_all", "xavier", "kaiming")
            dimension_interpolation: 维度插值方法 ("linear", "nearest", "cubic")
            validation_enabled: 是否启用迁移验证
            verbose: 是否输出详细信息
        """
        # 处理配置对象
        if isinstance(source_config, CWSAMScalingConfig):
            self.source_config = source_config.get_config()
        else:
            self.source_config = source_config
            
        if isinstance(target_config, CWSAMScalingConfig):
            self.target_config = target_config.get_config()
        else:
            self.target_config = target_config
        
        self.migration_strategy = migration_strategy
        self.expert_initialization = expert_initialization
        self.dimension_interpolation = dimension_interpolation
        self.validation_enabled = validation_enabled
        self.verbose = verbose
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 验证迁移可行性
        self._validate_migration_feasibility()
        
        # 计算迁移统计信息
        self.migration_stats = self._compute_migration_stats()
        
        if self.verbose:
            self._print_migration_summary()
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"WeightMigration_{id(self)}")
        logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _validate_migration_feasibility(self):
        """验证迁移可行性"""
        errors = []
        warnings_list = []
        
        # 检查模型规模兼容性
        source_params = self.source_config.estimate_parameters()
        target_params = self.target_config.estimate_parameters()
        
        if target_params <= source_params:
            warnings_list.append(
                f"目标模型参数量 ({target_params:,}) 不大于源模型 ({source_params:,})"
            )
        
        # 检查架构兼容性
        if self.source_config.patch_size != self.target_config.patch_size:
            errors.append("补丁大小不匹配，无法进行权重迁移")
        
        if self.source_config.img_size != self.target_config.img_size:
            warnings_list.append("图像大小不匹配，可能影响位置编码迁移")
        
        # 检查维度兼容性
        if self.target_config.embed_dim < self.source_config.embed_dim:
            errors.append("目标嵌入维度小于源嵌入维度，不支持降维迁移")
        
        if self.target_config.depth < self.source_config.depth:
            errors.append("目标模型深度小于源模型深度，不支持减层迁移")
        
        # 检查MoE兼容性
        if self.target_config.moe_num_experts < self.source_config.moe_num_experts:
            errors.append("目标专家数量小于源专家数量，不支持专家减少迁移")
        
        # 报告错误和警告
        if errors:
            error_msg = "权重迁移验证失败:\n" + "\n".join(f"- {e}" for e in errors)
            raise ValueError(error_msg)
        
        if warnings_list and self.verbose:
            warning_msg = "权重迁移警告:\n" + "\n".join(f"- {w}" for w in warnings_list)
            self.logger.warning(warning_msg)
    
    def _compute_migration_stats(self) -> Dict[str, Any]:
        """计算迁移统计信息"""
        stats = {
            "source_params": self.source_config.estimate_parameters(),
            "target_params": self.target_config.estimate_parameters(),
            "param_increase_ratio": 0.0,
            "depth_increase": self.target_config.depth - self.source_config.depth,
            "embed_dim_increase": self.target_config.embed_dim - self.source_config.embed_dim,
            "expert_increase": self.target_config.moe_num_experts - self.source_config.moe_num_experts,
            "moe_layer_change": (self.target_config.depth - self.target_config.moe_start_layer) - 
                               (self.source_config.depth - self.source_config.moe_start_layer),
            "requires_dimension_interpolation": self.target_config.embed_dim != self.source_config.embed_dim,
            "requires_expert_expansion": self.target_config.moe_num_experts != self.source_config.moe_num_experts,
            "requires_depth_expansion": self.target_config.depth != self.source_config.depth,
        }
        
        if stats["source_params"] > 0:
            stats["param_increase_ratio"] = stats["target_params"] / stats["source_params"]
        
        return stats
    
    def _print_migration_summary(self):
        """打印迁移摘要"""
        stats = self.migration_stats
        
        print(f"\n=== 权重迁移计划摘要 ===")
        print(f"源模型: {self.source_config.model_size} ({stats['source_params']:,} 参数)")
        print(f"目标模型: {self.target_config.model_size} ({stats['target_params']:,} 参数)")
        print(f"参数增长: {stats['param_increase_ratio']:.2f}x")
        print(f"深度变化: {self.source_config.depth} -> {self.target_config.depth} (+{stats['depth_increase']})")
        print(f"嵌入维度: {self.source_config.embed_dim} -> {self.target_config.embed_dim} (+{stats['embed_dim_increase']})")
        print(f"专家数量: {self.source_config.moe_num_experts} -> {self.target_config.moe_num_experts} (+{stats['expert_increase']})")
        print(f"MoE层变化: +{stats['moe_layer_change']}")
        print(f"迁移策略: {self.migration_strategy}")
        print(f"专家初始化: {self.expert_initialization}")
        print(f"维度插值: {self.dimension_interpolation}")
        print("=" * 30)
    
    def migrate_weights(self, source_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        执行权重迁移
        
        Args:
            source_state_dict: 源模型的状态字典
            
        Returns:
            迁移后的目标模型状态字典
        """
        self.logger.info("开始权重迁移...")
        
        # 创建目标状态字典
        target_state_dict = OrderedDict()
        
        # 分类处理不同类型的权重
        migration_plan = self._create_migration_plan(source_state_dict)
        
        # 1. 迁移基础组件权重
        self._migrate_basic_weights(source_state_dict, target_state_dict, migration_plan)
        
        # 2. 处理Transformer块权重
        self._migrate_transformer_blocks(source_state_dict, target_state_dict, migration_plan)
        
        # 3. 处理MoE专家权重
        self._migrate_moe_weights(source_state_dict, target_state_dict, migration_plan)
        
        # 4. 处理位置编码
        self._migrate_position_embeddings(source_state_dict, target_state_dict, migration_plan)
        
        # 5. 处理解码器权重
        self._migrate_decoder_weights(source_state_dict, target_state_dict, migration_plan)
        
        # 6. 处理多尺度特征模块
        self._migrate_multi_scale_weights(source_state_dict, target_state_dict, migration_plan)
        
        # 验证迁移结果
        if self.validation_enabled:
            self._validate_migrated_weights(target_state_dict)
        
        self.logger.info("权重迁移完成")
        return target_state_dict
    
    def _create_migration_plan(self, source_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """创建详细的迁移计划"""
        plan = {
            "basic_weights": [],
            "transformer_blocks": {},
            "moe_weights": {},
            "position_embeddings": [],
            "decoder_weights": [],
            "multi_scale_weights": [],
            "dimension_mappings": {},
            "expert_mappings": {}
        }
        
        # 分析源权重结构
        for key in source_state_dict.keys():
            if "patch_embed" in key or "neck" in key:
                plan["basic_weights"].append(key)
            elif "blocks." in key:
                block_idx = self._extract_block_index(key)
                if block_idx not in plan["transformer_blocks"]:
                    plan["transformer_blocks"][block_idx] = []
                plan["transformer_blocks"][block_idx].append(key)
                
                # 检查是否为MoE权重
                if "experts" in key or "gate" in key:
                    if block_idx not in plan["moe_weights"]:
                        plan["moe_weights"][block_idx] = []
                    plan["moe_weights"][block_idx].append(key)
            elif "pos_embed" in key:
                plan["position_embeddings"].append(key)
            elif "transformer" in key or "iou_" in key or "mask_" in key:
                plan["decoder_weights"].append(key)
            elif "multi_scale" in key:
                plan["multi_scale_weights"].append(key)
        
        # 创建维度映射
        plan["dimension_mappings"] = self._create_dimension_mappings()
        
        # 创建专家映射
        plan["expert_mappings"] = self._create_expert_mappings()
        
        return plan
    
    def _extract_block_index(self, key: str) -> int:
        """从权重键中提取块索引"""
        import re
        match = re.search(r'blocks\.(\d+)\.', key)
        return int(match.group(1)) if match else -1
    
    def _create_dimension_mappings(self) -> Dict[str, Any]:
        """创建维度映射关系"""
        source_dim = self.source_config.embed_dim
        target_dim = self.target_config.embed_dim
        
        if source_dim == target_dim:
            return {"type": "identity"}
        
        return {
            "type": "interpolation",
            "method": self.dimension_interpolation,
            "source_dim": source_dim,
            "target_dim": target_dim,
            "scale_factor": target_dim / source_dim
        }
    
    def _create_expert_mappings(self) -> Dict[str, Any]:
        """创建专家映射关系"""
        source_experts = self.source_config.moe_num_experts
        target_experts = self.target_config.moe_num_experts
        
        if source_experts == target_experts:
            return {"type": "identity"}
        
        # 创建专家复制策略
        if self.expert_initialization == "copy_random":
            # 随机选择源专家进行复制
            expert_map = {}
            for i in range(target_experts):
                source_idx = i % source_experts
                expert_map[i] = source_idx
        elif self.expert_initialization == "copy_all":
            # 复制所有源专家，然后随机初始化新专家
            expert_map = {}
            for i in range(min(source_experts, target_experts)):
                expert_map[i] = i
            # 新专家将使用随机初始化
        else:
            expert_map = {}
        
        return {
            "type": "expansion",
            "method": self.expert_initialization,
            "source_experts": source_experts,
            "target_experts": target_experts,
            "expert_map": expert_map
        }
    
    def _migrate_basic_weights(
        self, 
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移基础组件权重"""
        self.logger.info("迁移基础组件权重...")
        
        for key in migration_plan["basic_weights"]:
            if key not in source_state_dict:
                continue
                
            source_weight = source_state_dict[key]
            
            # 处理维度变化
            if self._requires_dimension_adjustment(key, source_weight):
                target_weight = self._adjust_weight_dimensions(
                    source_weight, key, migration_plan["dimension_mappings"]
                )
            else:
                target_weight = source_weight.clone()
            
            target_state_dict[key] = target_weight
            
        self.logger.info(f"迁移了 {len(migration_plan['basic_weights'])} 个基础权重")
    
    def _migrate_transformer_blocks(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移Transformer块权重"""
        self.logger.info("迁移Transformer块权重...")
        
        source_depth = self.source_config.depth
        target_depth = self.target_config.depth
        
        migrated_blocks = 0
        
        # 迁移现有块
        for block_idx in range(min(source_depth, target_depth)):
            if block_idx in migration_plan["transformer_blocks"]:
                for key in migration_plan["transformer_blocks"][block_idx]:
                    if key in source_state_dict:
                        source_weight = source_state_dict[key]
                        
                        # 跳过MoE权重，单独处理
                        if "experts" in key or "gate" in key:
                            continue
                        
                        # 处理维度变化
                        if self._requires_dimension_adjustment(key, source_weight):
                            target_weight = self._adjust_weight_dimensions(
                                source_weight, key, migration_plan["dimension_mappings"]
                            )
                        else:
                            target_weight = source_weight.clone()
                        
                        target_state_dict[key] = target_weight
                        migrated_blocks += 1
        
        # 初始化新增的块
        if target_depth > source_depth:
            self._initialize_new_transformer_blocks(
                target_state_dict, source_depth, target_depth, migration_plan
            )
        
        self.logger.info(f"迁移了 {migrated_blocks} 个Transformer块权重")
    
    def _migrate_moe_weights(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移MoE专家权重"""
        self.logger.info("迁移MoE专家权重...")
        
        expert_mappings = migration_plan["expert_mappings"]
        migrated_experts = 0
        
        for block_idx, moe_keys in migration_plan["moe_weights"].items():
            # 检查目标块是否应该有MoE
            target_block_idx = block_idx
            if target_block_idx >= self.target_config.moe_start_layer:
                
                for key in moe_keys:
                    if key not in source_state_dict:
                        continue
                    
                    source_weight = source_state_dict[key]
                    
                    if "experts" in key:
                        # 处理专家权重
                        target_weight = self._migrate_expert_weights(
                            source_weight, key, expert_mappings, migration_plan["dimension_mappings"]
                        )
                    elif "gate" in key:
                        # 处理门控权重
                        target_weight = self._migrate_gate_weights(
                            source_weight, key, expert_mappings, migration_plan["dimension_mappings"]
                        )
                    else:
                        # 其他MoE相关权重
                        if self._requires_dimension_adjustment(key, source_weight):
                            target_weight = self._adjust_weight_dimensions(
                                source_weight, key, migration_plan["dimension_mappings"]
                            )
                        else:
                            target_weight = source_weight.clone()
                    
                    target_state_dict[key] = target_weight
                    migrated_experts += 1
        
        # 为新增的MoE层初始化专家
        self._initialize_new_moe_layers(target_state_dict, migration_plan)
        
        self.logger.info(f"迁移了 {migrated_experts} 个MoE权重")
    
    def _migrate_expert_weights(
        self,
        source_weight: torch.Tensor,
        key: str,
        expert_mappings: Dict[str, Any],
        dimension_mappings: Dict[str, Any]
    ) -> torch.Tensor:
        """迁移专家权重"""
        if expert_mappings["type"] == "identity":
            # 专家数量相同，直接复制
            if self._requires_dimension_adjustment(key, source_weight):
                return self._adjust_weight_dimensions(source_weight, key, dimension_mappings)
            else:
                return source_weight.clone()
        
        # 专家扩展情况
        source_experts = expert_mappings["source_experts"]
        target_experts = expert_mappings["target_experts"]
        expert_map = expert_mappings["expert_map"]
        
        # 解析专家索引
        expert_idx = self._extract_expert_index(key)
        
        if expert_idx < source_experts:
            # 现有专家，直接迁移
            target_weight = source_weight.clone()
        else:
            # 新专家，根据策略初始化
            if expert_mappings["method"] == "copy_random" and expert_idx in expert_map:
                # 复制指定的源专家
                target_weight = source_weight.clone()  # 简化处理
            else:
                # 随机初始化
                target_weight = self._initialize_expert_weight(source_weight, key)
        
        # 处理维度变化
        if self._requires_dimension_adjustment(key, target_weight):
            target_weight = self._adjust_weight_dimensions(target_weight, key, dimension_mappings)
        
        return target_weight
    
    def _migrate_gate_weights(
        self,
        source_weight: torch.Tensor,
        key: str,
        expert_mappings: Dict[str, Any],
        dimension_mappings: Dict[str, Any]
    ) -> torch.Tensor:
        """迁移门控权重"""
        if expert_mappings["type"] == "identity":
            # 专家数量相同
            if self._requires_dimension_adjustment(key, source_weight):
                return self._adjust_weight_dimensions(source_weight, key, dimension_mappings)
            else:
                return source_weight.clone()
        
        # 专家数量变化，需要调整门控权重
        source_experts = expert_mappings["source_experts"]
        target_experts = expert_mappings["target_experts"]
        
        # 门控权重通常是 [embed_dim, num_experts] 或 [num_experts]
        if source_weight.dim() == 2:
            # 权重矩阵
            embed_dim = source_weight.shape[0]
            target_weight = torch.zeros(embed_dim, target_experts, dtype=source_weight.dtype, device=source_weight.device)
            
            # 复制现有专家的门控权重
            target_weight[:, :source_experts] = source_weight
            
            # 为新专家初始化门控权重
            if target_experts > source_experts:
                nn.init.normal_(target_weight[:, source_experts:], mean=0.0, std=0.02)
                
        elif source_weight.dim() == 1:
            # 偏置向量
            target_weight = torch.zeros(target_experts, dtype=source_weight.dtype, device=source_weight.device)
            target_weight[:source_experts] = source_weight
            # 新专家的偏置保持为0
        else:
            raise ValueError(f"不支持的门控权重维度: {source_weight.shape}")
        
        # 处理嵌入维度变化
        if self._requires_dimension_adjustment(key, target_weight):
            target_weight = self._adjust_weight_dimensions(target_weight, key, dimension_mappings)
        
        return target_weight
    
    def _migrate_position_embeddings(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移位置编码"""
        self.logger.info("迁移位置编码...")
        
        for key in migration_plan["position_embeddings"]:
            if key not in source_state_dict:
                continue
            
            source_pos_embed = source_state_dict[key]
            
            # 处理维度变化
            if self._requires_dimension_adjustment(key, source_pos_embed):
                target_pos_embed = self._adjust_weight_dimensions(
                    source_pos_embed, key, migration_plan["dimension_mappings"]
                )
            else:
                target_pos_embed = source_pos_embed.clone()
            
            target_state_dict[key] = target_pos_embed
        
        self.logger.info(f"迁移了 {len(migration_plan['position_embeddings'])} 个位置编码")
    
    def _migrate_decoder_weights(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移解码器权重"""
        self.logger.info("迁移解码器权重...")
        
        migrated_decoder_weights = 0
        
        for key in migration_plan["decoder_weights"]:
            if key not in source_state_dict:
                continue
            
            source_weight = source_state_dict[key]
            
            # 处理解码器深度变化
            if "transformer.layers" in key:
                layer_idx = self._extract_decoder_layer_index(key)
                if layer_idx >= self.target_config.decoder_depth:
                    continue  # 跳过超出目标深度的层
            
            # 处理维度变化
            if self._requires_dimension_adjustment(key, source_weight):
                target_weight = self._adjust_weight_dimensions(
                    source_weight, key, migration_plan["dimension_mappings"]
                )
            else:
                target_weight = source_weight.clone()
            
            target_state_dict[key] = target_weight
            migrated_decoder_weights += 1
        
        # 初始化新增的解码器层
        if self.target_config.decoder_depth > self.source_config.decoder_depth:
            self._initialize_new_decoder_layers(target_state_dict, migration_plan)
        
        self.logger.info(f"迁移了 {migrated_decoder_weights} 个解码器权重")
    
    def _migrate_multi_scale_weights(
        self,
        source_state_dict: Dict[str, torch.Tensor],
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """迁移多尺度特征模块权重"""
        if not self.target_config.use_multi_scale:
            return
        
        self.logger.info("迁移多尺度特征模块权重...")
        
        migrated_multi_scale_weights = 0
        
        for key in migration_plan["multi_scale_weights"]:
            if key not in source_state_dict:
                continue
            
            source_weight = source_state_dict[key]
            
            # 处理维度变化
            if self._requires_dimension_adjustment(key, source_weight):
                target_weight = self._adjust_weight_dimensions(
                    source_weight, key, migration_plan["dimension_mappings"]
                )
            else:
                target_weight = source_weight.clone()
            
            target_state_dict[key] = target_weight
            migrated_multi_scale_weights += 1
        
        # 如果源模型没有多尺度模块但目标模型需要，则初始化
        if not self.source_config.use_multi_scale and self.target_config.use_multi_scale:
            self._initialize_multi_scale_module(target_state_dict, migration_plan)
        
        self.logger.info(f"迁移了 {migrated_multi_scale_weights} 个多尺度权重")
    
    def _requires_dimension_adjustment(self, key: str, weight: torch.Tensor) -> bool:
        """检查权重是否需要维度调整"""
        if self.source_config.embed_dim == self.target_config.embed_dim:
            return False
        
        # 检查权重是否包含嵌入维度
        embed_dim_patterns = [
            "embed", "proj", "norm", "attn", "mlp", "linear"
        ]
        
        return any(pattern in key.lower() for pattern in embed_dim_patterns)
    
    def _adjust_weight_dimensions(
        self,
        source_weight: torch.Tensor,
        key: str,
        dimension_mappings: Dict[str, Any]
    ) -> torch.Tensor:
        """调整权重维度"""
        if dimension_mappings["type"] == "identity":
            return source_weight.clone()
        
        source_dim = dimension_mappings["source_dim"]
        target_dim = dimension_mappings["target_dim"]
        method = dimension_mappings["method"]
        
        # 确定需要调整的维度
        weight_shape = list(source_weight.shape)
        dim_to_adjust = None
        
        for i, dim_size in enumerate(weight_shape):
            if dim_size == source_dim:
                dim_to_adjust = i
                break
        
        if dim_to_adjust is None:
            # 没有找到匹配的维度，直接返回
            return source_weight.clone()
        
        # 执行维度调整
        if method == "linear":
            target_weight = self._linear_interpolation(
                source_weight, dim_to_adjust, target_dim
            )
        elif method == "nearest":
            target_weight = self._nearest_interpolation(
                source_weight, dim_to_adjust, target_dim
            )
        elif method == "cubic":
            target_weight = self._cubic_interpolation(
                source_weight, dim_to_adjust, target_dim
            )
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        return target_weight
    
    def _linear_interpolation(
        self, 
        weight: torch.Tensor, 
        dim: int, 
        target_size: int
    ) -> torch.Tensor:
        """线性插值调整权重维度"""
        if dim == 0:
            # 第一维度插值
            indices = torch.linspace(0, weight.shape[0] - 1, target_size)
            indices_floor = indices.floor().long()
            indices_ceil = indices.ceil().long()
            alpha = indices - indices_floor.float()
            
            weight_floor = weight[indices_floor]
            weight_ceil = weight[indices_ceil]
            
            # 处理边界情况
            alpha = alpha.view(-1, *([1] * (weight.dim() - 1)))
            interpolated = weight_floor * (1 - alpha) + weight_ceil * alpha
            
        elif dim == 1 and weight.dim() >= 2:
            # 第二维度插值
            indices = torch.linspace(0, weight.shape[1] - 1, target_size)
            indices_floor = indices.floor().long()
            indices_ceil = indices.ceil().long()
            alpha = indices - indices_floor.float()
            
            weight_floor = weight[:, indices_floor]
            weight_ceil = weight[:, indices_ceil]
            
            alpha = alpha.view(1, -1, *([1] * (weight.dim() - 2)))
            interpolated = weight_floor * (1 - alpha) + weight_ceil * alpha
            
        else:
            # 其他维度，使用简单的重复或截断
            if target_size > weight.shape[dim]:
                # 扩展：重复最后的元素
                repeat_times = [1] * weight.dim()
                repeat_times[dim] = target_size - weight.shape[dim]
                
                last_slice = [slice(None)] * weight.dim()
                last_slice[dim] = slice(-1, None)
                
                repeated_part = weight[tuple(last_slice)].repeat(repeat_times)
                interpolated = torch.cat([weight, repeated_part], dim=dim)
            else:
                # 截断
                slice_obj = [slice(None)] * weight.dim()
                slice_obj[dim] = slice(0, target_size)
                interpolated = weight[tuple(slice_obj)]
        
        return interpolated
    
    def _nearest_interpolation(
        self, 
        weight: torch.Tensor, 
        dim: int, 
        target_size: int
    ) -> torch.Tensor:
        """最近邻插值调整权重维度"""
        source_size = weight.shape[dim]
        indices = torch.round(torch.linspace(0, source_size - 1, target_size)).long()
        
        # 使用索引选择
        return torch.index_select(weight, dim, indices)
    
    def _cubic_interpolation(
        self, 
        weight: torch.Tensor, 
        dim: int, 
        target_size: int
    ) -> torch.Tensor:
        """三次插值调整权重维度（简化版本）"""
        # 对于权重迁移，三次插值可能过于复杂，这里使用线性插值作为替代
        return self._linear_interpolation(weight, dim, target_size)
    
    def _extract_expert_index(self, key: str) -> int:
        """从权重键中提取专家索引"""
        import re
        match = re.search(r'experts\.(\d+)\.', key)
        return int(match.group(1)) if match else -1
    
    def _extract_decoder_layer_index(self, key: str) -> int:
        """从权重键中提取解码器层索引"""
        import re
        match = re.search(r'layers\.(\d+)\.', key)
        return int(match.group(1)) if match else -1
    
    def _initialize_expert_weight(self, reference_weight: torch.Tensor, key: str) -> torch.Tensor:
        """初始化新专家权重"""
        new_weight = torch.empty_like(reference_weight)
        
        if "linear" in key.lower() or "weight" in key:
            # 线性层权重使用Xavier初始化
            nn.init.xavier_uniform_(new_weight)
        elif "bias" in key:
            # 偏置初始化为0
            nn.init.zeros_(new_weight)
        else:
            # 其他权重使用正态分布初始化
            nn.init.normal_(new_weight, mean=0.0, std=0.02)
        
        return new_weight
    
    def _initialize_new_transformer_blocks(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        source_depth: int,
        target_depth: int,
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的Transformer块"""
        self.logger.info(f"初始化新增的Transformer块 ({source_depth} -> {target_depth})...")
        
        # 获取参考块的权重结构
        reference_block_idx = source_depth - 1
        if reference_block_idx in migration_plan["transformer_blocks"]:
            reference_keys = migration_plan["transformer_blocks"][reference_block_idx]
            
            for new_block_idx in range(source_depth, target_depth):
                for ref_key in reference_keys:
                    # 跳过MoE权重，单独处理
                    if "experts" in ref_key or "gate" in ref_key:
                        continue
                    
                    # 生成新块的键名
                    new_key = ref_key.replace(f"blocks.{reference_block_idx}.", f"blocks.{new_block_idx}.")
                    
                    # 获取参考权重
                    if ref_key in target_state_dict:
                        ref_weight = target_state_dict[ref_key]
                        new_weight = self._initialize_layer_weight(ref_weight, new_key)
                        target_state_dict[new_key] = new_weight
    
    def _initialize_new_moe_layers(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的MoE层"""
        self.logger.info("初始化新增的MoE层...")
        
        target_depth = self.target_config.depth
        target_moe_start = self.target_config.moe_start_layer
        source_moe_start = self.source_config.moe_start_layer
        
        # 为新的MoE层初始化专家
        for block_idx in range(target_moe_start, target_depth):
            if block_idx < source_moe_start:
                # 这是新的MoE层，需要完全初始化
                self._initialize_moe_block(target_state_dict, block_idx, migration_plan)
    
    def _initialize_moe_block(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        block_idx: int,
        migration_plan: Dict[str, Any]
    ):
        """初始化单个MoE块"""
        embed_dim = self.target_config.embed_dim
        mlp_dim = embed_dim * self.target_config.mlp_ratio
        num_experts = self.target_config.moe_num_experts
        
        # 初始化专家权重
        for expert_idx in range(num_experts):
            # 专家的第一个线性层
            key1 = f"blocks.{block_idx}.mlp.experts.{expert_idx}.0.weight"
            weight1 = torch.empty(mlp_dim, embed_dim)
            nn.init.xavier_uniform_(weight1)
            target_state_dict[key1] = weight1
            
            bias1_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.0.bias"
            bias1 = torch.zeros(mlp_dim)
            target_state_dict[bias1_key] = bias1
            
            # 专家的第二个线性层
            key2 = f"blocks.{block_idx}.mlp.experts.{expert_idx}.2.weight"
            weight2 = torch.empty(embed_dim, mlp_dim)
            nn.init.xavier_uniform_(weight2)
            target_state_dict[key2] = weight2
            
            bias2_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.2.bias"
            bias2 = torch.zeros(embed_dim)
            target_state_dict[bias2_key] = bias2
        
        # 初始化门控网络
        gate_key = f"blocks.{block_idx}.mlp.gate.weight"
        gate_weight = torch.empty(num_experts, embed_dim)
        nn.init.normal_(gate_weight, mean=0.0, std=0.02)
        target_state_dict[gate_key] = gate_weight
        
        gate_bias_key = f"blocks.{block_idx}.mlp.gate.bias"
        gate_bias = torch.zeros(num_experts)
        target_state_dict[gate_bias_key] = gate_bias
    
    def _initialize_new_decoder_layers(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的解码器层"""
        source_depth = self.source_config.decoder_depth
        target_depth = self.target_config.decoder_depth
        
        if target_depth <= source_depth:
            return
        
        self.logger.info(f"初始化新增的解码器层 ({source_depth} -> {target_depth})...")
        
        # 获取参考层的权重结构
        reference_keys = [key for key in migration_plan["decoder_weights"] 
                         if f"layers.{source_depth-1}." in key]
        
        for new_layer_idx in range(source_depth, target_depth):
            for ref_key in reference_keys:
                new_key = ref_key.replace(f"layers.{source_depth-1}.", f"layers.{new_layer_idx}.")
                
                # 获取参考权重维度
                ref_key_in_dict = None
                for existing_key in target_state_dict.keys():
                    if ref_key in existing_key:
                        ref_key_in_dict = existing_key
                        break
                
                if ref_key_in_dict:
                    ref_weight = target_state_dict[ref_key_in_dict]
                    new_weight = self._initialize_layer_weight(ref_weight, new_key)
                    target_state_dict[new_key] = new_weight
    
    def _initialize_multi_scale_module(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化多尺度特征模块"""
        self.logger.info("初始化多尺度特征模块...")
        
        embed_dim = self.target_config.embed_dim
        scales = [1, 2, 4]  # 默认尺度
        
        # 初始化尺度卷积层
        for i, scale in enumerate(scales):
            conv_weight_key = f"multi_scale.scale_convs.{i}.weight"
            conv_weight = torch.empty(embed_dim, embed_dim, 3, 3)
            nn.init.xavier_uniform_(conv_weight)
            target_state_dict[conv_weight_key] = conv_weight
            
            conv_bias_key = f"multi_scale.scale_convs.{i}.bias"
            conv_bias = torch.zeros(embed_dim)
            target_state_dict[conv_bias_key] = conv_bias
        
        # 初始化融合卷积层
        fusion_weight_key = "multi_scale.fusion_conv.weight"
        fusion_weight = torch.empty(embed_dim, embed_dim * len(scales), 1, 1)
        nn.init.xavier_uniform_(fusion_weight)
        target_state_dict[fusion_weight_key] = fusion_weight
        
        fusion_bias_key = "multi_scale.fusion_conv.bias"
        fusion_bias = torch.zeros(embed_dim)
        target_state_dict[fusion_bias_key] = fusion_bias
        
        # 初始化上采样卷积层
        for i, scale in enumerate(scales[1:]):  # 跳过scale=1
            upsample_weight_key = f"multi_scale.upsample_convs.{i}.weight"
            upsample_weight = torch.empty(embed_dim, embed_dim, scale*2, scale*2)
            nn.init.xavier_uniform_(upsample_weight)
            target_state_dict[upsample_weight_key] = upsample_weight
            
            upsample_bias_key = f"multi_scale.upsample_convs.{i}.bias"
            upsample_bias = torch.zeros(embed_dim)
            target_state_dict[upsample_bias_key] = upsample_bias
    
    def _initialize_layer_weight(self, reference_weight: torch.Tensor, key: str) -> torch.Tensor:
        """初始化层权重"""
        new_weight = torch.empty_like(reference_weight)
        
        if "weight" in key and reference_weight.dim() >= 2:
            # 线性层或卷积层权重
            nn.init.xavier_uniform_(new_weight)
        elif "bias" in key:
            # 偏置
            nn.init.zeros_(new_weight)
        elif "norm" in key:
            # 归一化层
            if "weight" in key:
                nn.init.ones_(new_weight)
            else:
                nn.init.zeros_(new_weight)
        else:
            # 其他权重
            nn.init.normal_(new_weight, mean=0.0, std=0.02)
        
        return new_weight
    
    def _validate_migrated_weights(self, target_state_dict: Dict[str, torch.Tensor]):
        """验证迁移后的权重"""
        self.logger.info("验证迁移后的权重...")
        
        validation_errors = []
        
        # 检查权重形状
        for key, weight in target_state_dict.items():
            if torch.isnan(weight).any():
                validation_errors.append(f"权重 {key} 包含NaN值")
            
            if torch.isinf(weight).any():
                validation_errors.append(f"权重 {key} 包含无穷值")
            
            # 检查权重范围
            if weight.abs().max() > 100:
                validation_errors.append(f"权重 {key} 值过大: {weight.abs().max()}")
        
        # 检查关键组件
        required_components = [
            "patch_embed",
            "pos_embed", 
            "blocks.0.",
            "neck"
        ]
        
        for component in required_components:
            if not any(component in key for key in target_state_dict.keys()):
                validation_errors.append(f"缺少关键组件: {component}")
        
        if validation_errors:
            error_msg = "权重验证失败:\n" + "\n".join(f"- {e}" for e in validation_errors)
            raise ValueError(error_msg)
        
        self.logger.info("权重验证通过")
    
    def get_migration_report(self) -> Dict[str, Any]:
        """获取迁移报告"""
        return {
            "migration_stats": self.migration_stats,
            "source_config": self.source_config.__dict__ if hasattr(self.source_config, '__dict__') else self.source_config,
            "target_config": self.target_config.__dict__ if hasattr(self.target_config, '__dict__') else self.target_config,
            "migration_strategy": self.migration_strategy,
            "expert_initialization": self.expert_initialization,
            "dimension_interpolation": self.dimension_interpolation
        }
    
    def save_migration_report(self, filepath: str):
        """保存迁移报告"""
        import json
        
        report = self.get_migration_report()
        
        # 转换不可序列化的对象
        def convert_for_json(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif hasattr(obj, '__dict__'):
                return obj.__dict__
            else:
                return str(obj)
        
        # 递归转换
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                try:
                    json.dumps(obj)
                    return obj
                except:
                    return convert_for_json(obj)
        
        serializable_report = recursive_convert(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"迁移报告已保存到: {filepath}")


def create_weight_migration_manager(
    source_model_size: str,
    target_model_size: str,
    migration_strategy: str = "adaptive",
    expert_initialization: str = "copy_random",
    dimension_interpolation: str = "linear",
    validation_enabled: bool = True,
    verbose: bool = True
) -> WeightMigrationManager:
    """
    创建权重迁移管理器的便捷函数
    
    Args:
        source_model_size: 源模型规模
        target_model_size: 目标模型规模
        migration_strategy: 迁移策略
        expert_initialization: 专家初始化策略
        dimension_interpolation: 维度插值方法
        validation_enabled: 是否启用验证
        verbose: 是否输出详细信息
    
    Returns:
        WeightMigrationManager: 权重迁移管理器实例
    """
    source_config = CWSAMScalingConfig(source_model_size)
    target_config = CWSAMScalingConfig(target_model_size)
    
    return WeightMigrationManager(
        source_config=source_config,
        target_config=target_config,
        migration_strategy=migration_strategy,
        expert_initialization=expert_initialization,
        dimension_interpolation=dimension_interpolation,
        validation_enabled=validation_enabled,
        verbose=verbose
    )


if __name__ == "__main__":
    # 测试权重迁移管理器
    print("=" * 50)
    print("测试权重迁移管理器")
    
    try:
        # 创建权重迁移管理器
        migration_manager = create_weight_migration_manager(
            source_model_size="1.5B",
            target_model_size="3B",
            verbose=True
        )
        
        print("✓ 权重迁移管理器创建成功")
        
        # 创建模拟的源权重
        source_state_dict = {
            "patch_embed.proj.weight": torch.randn(1280, 3, 16, 16),
            "patch_embed.proj.bias": torch.randn(1280),
            "pos_embed": torch.randn(1, 64, 64, 1280),
            "blocks.0.norm1.weight": torch.randn(1280),
            "blocks.0.norm1.bias": torch.randn(1280),
            "blocks.0.attn.qkv.weight": torch.randn(3840, 1280),
            "blocks.0.attn.qkv.bias": torch.randn(3840),
            "blocks.0.attn.proj.weight": torch.randn(1280, 1280),
            "blocks.0.attn.proj.bias": torch.randn(1280),
            "blocks.28.mlp.experts.0.0.weight": torch.randn(5120, 1280),
            "blocks.28.mlp.experts.0.0.bias": torch.randn(5120),
            "blocks.28.mlp.experts.0.2.weight": torch.randn(1280, 5120),
            "blocks.28.mlp.experts.0.2.bias": torch.randn(1280),
            "blocks.28.mlp.gate.weight": torch.randn(16, 1280),
            "blocks.28.mlp.gate.bias": torch.randn(16),
            "neck.0.weight": torch.randn(256, 1280),
            "neck.0.bias": torch.randn(256)
        }
        
        print(f"源权重数量: {len(source_state_dict)}")
        
        # 执行权重迁移
        target_state_dict = migration_manager.migrate_weights(source_state_dict)
        
        print(f"✓ 迁移成功，目标权重数量: {len(target_state_dict)}")
        
        # 获取迁移报告
        report = migration_manager.get_migration_report()
        print(f"参数增长: {report['migration_stats']['param_increase_ratio']:.2f}x")
        
    except Exception as e:
        print(f"✗ 迁移失败: {e}")
    
    def _initialize_new_transformer_blocks(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        source_depth: int,
        target_depth: int,
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的Transformer块"""
        self.logger.info(f"初始化 {target_depth - source_depth} 个新Transformer块...")
        
        # 使用最后一个源块作为参考
        reference_block_idx = source_depth - 1
        
        for new_block_idx in range(source_depth, target_depth):
            # 复制参考块的结构
            if reference_block_idx in migration_plan["transformer_blocks"]:
                for ref_key in migration_plan["transformer_blocks"][reference_block_idx]:
                    # 创建新块的键
                    new_key = ref_key.replace(f"blocks.{reference_block_idx}.", f"blocks.{new_block_idx}.")
                    
                    # 跳过MoE权重，单独处理
                    if "experts" in ref_key or "gate" in ref_key:
                        continue
                    
                    # 获取参考权重
                    if ref_key in target_state_dict:
                        ref_weight = target_state_dict[ref_key]
                        
                        # 初始化新权重
                        new_weight = self._initialize_transformer_weight(ref_weight, new_key)
                        target_state_dict[new_key] = new_weight
    
    def _initialize_new_moe_layers(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的MoE层"""
        source_moe_start = self.source_config.moe_start_layer
        target_moe_start = self.target_config.moe_start_layer
        
        if target_moe_start < source_moe_start:
            # 需要为更早的层添加MoE
            self.logger.info(f"为层 {target_moe_start} 到 {source_moe_start-1} 初始化MoE...")
            
            # 使用第一个现有MoE层作为参考
            reference_moe_block = source_moe_start
            
            for block_idx in range(target_moe_start, source_moe_start):
                self._initialize_moe_for_block(
                    target_state_dict, block_idx, reference_moe_block, migration_plan
                )
    
    def _initialize_new_decoder_layers(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化新增的解码器层"""
        source_depth = self.source_config.decoder_depth
        target_depth = self.target_config.decoder_depth
        
        if target_depth > source_depth:
            self.logger.info(f"初始化 {target_depth - source_depth} 个新解码器层...")
            
            # 使用最后一个源层作为参考
            reference_layer_idx = source_depth - 1
            
            for new_layer_idx in range(source_depth, target_depth):
                self._initialize_decoder_layer(
                    target_state_dict, new_layer_idx, reference_layer_idx, migration_plan
                )
    
    def _initialize_multi_scale_module(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        migration_plan: Dict[str, Any]
    ):
        """初始化多尺度特征模块"""
        self.logger.info("初始化多尺度特征模块...")
        
        embed_dim = self.target_config.embed_dim
        
        # 初始化多尺度卷积权重
        scales = [1, 2, 4]
        for i, scale in enumerate(scales):
            # 尺度卷积
            conv_weight_key = f"multi_scale_pyramid.scale_convs.{i}.weight"
            conv_bias_key = f"multi_scale_pyramid.scale_convs.{i}.bias"
            
            conv_weight = torch.empty(embed_dim, embed_dim, 3, 3)
            nn.init.kaiming_normal_(conv_weight, mode='fan_out', nonlinearity='relu')
            target_state_dict[conv_weight_key] = conv_weight
            
            conv_bias = torch.zeros(embed_dim)
            target_state_dict[conv_bias_key] = conv_bias
        
        # 融合卷积
        fusion_weight_key = "multi_scale_pyramid.fusion_conv.weight"
        fusion_bias_key = "multi_scale_pyramid.fusion_conv.bias"
        
        fusion_weight = torch.empty(embed_dim, embed_dim * len(scales), 1, 1)
        nn.init.kaiming_normal_(fusion_weight, mode='fan_out', nonlinearity='relu')
        target_state_dict[fusion_weight_key] = fusion_weight
        
        fusion_bias = torch.zeros(embed_dim)
        target_state_dict[fusion_bias_key] = fusion_bias
    
    def _initialize_transformer_weight(self, reference_weight: torch.Tensor, key: str) -> torch.Tensor:
        """初始化Transformer权重"""
        new_weight = torch.empty_like(reference_weight)
        
        if "norm" in key and "weight" in key:
            nn.init.ones_(new_weight)
        elif "norm" in key and "bias" in key:
            nn.init.zeros_(new_weight)
        elif "attn" in key and "weight" in key:
            nn.init.xavier_uniform_(new_weight)
        elif "mlp" in key and "weight" in key:
            nn.init.xavier_uniform_(new_weight)
        elif "bias" in key:
            nn.init.zeros_(new_weight)
        else:
            nn.init.normal_(new_weight, mean=0.0, std=0.02)
        
        return new_weight
    
    def _initialize_moe_for_block(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        block_idx: int,
        reference_block_idx: int,
        migration_plan: Dict[str, Any]
    ):
        """为指定块初始化MoE"""
        embed_dim = self.target_config.embed_dim
        num_experts = self.target_config.moe_num_experts
        mlp_ratio = self.target_config.mlp_ratio
        mlp_dim = int(embed_dim * mlp_ratio)
        
        # 初始化专家网络权重
        for expert_idx in range(num_experts):
            # 专家MLP第一层
            expert_w1_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.0.weight"
            expert_b1_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.0.bias"
            
            w1 = torch.empty(mlp_dim, embed_dim)
            nn.init.xavier_uniform_(w1)
            target_state_dict[expert_w1_key] = w1
            
            b1 = torch.zeros(mlp_dim)
            target_state_dict[expert_b1_key] = b1
            
            # 专家MLP第二层
            expert_w2_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.2.weight"
            expert_b2_key = f"blocks.{block_idx}.mlp.experts.{expert_idx}.2.bias"
            
            w2 = torch.empty(embed_dim, mlp_dim)
            nn.init.xavier_uniform_(w2)
            target_state_dict[expert_w2_key] = w2
            
            b2 = torch.zeros(embed_dim)
            target_state_dict[expert_b2_key] = b2
        
        # 初始化门控网络权重
        gate_weight_key = f"blocks.{block_idx}.mlp.gate.weight"
        gate_bias_key = f"blocks.{block_idx}.mlp.gate.bias"
        
        gate_weight = torch.empty(num_experts, embed_dim)
        nn.init.normal_(gate_weight, mean=0.0, std=0.02)
        target_state_dict[gate_weight_key] = gate_weight
        
        gate_bias = torch.zeros(num_experts)
        target_state_dict[gate_bias_key] = gate_bias
    
    def _initialize_decoder_layer(
        self,
        target_state_dict: Dict[str, torch.Tensor],
        layer_idx: int,
        reference_layer_idx: int,
        migration_plan: Dict[str, Any]
    ):
        """初始化解码器层"""
        transformer_dim = 256  # 默认解码器transformer维度
        
        # 自注意力层
        self_attn_keys = [
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.q_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.k_proj.weight", 
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.v_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.out_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.q_proj.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.k_proj.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.v_proj.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.self_attn.out_proj.bias"
        ]
        
        for key in self_attn_keys:
            if "weight" in key:
                weight = torch.empty(transformer_dim, transformer_dim)
                nn.init.xavier_uniform_(weight)
                target_state_dict[key] = weight
            else:  # bias
                bias = torch.zeros(transformer_dim)
                target_state_dict[key] = bias
        
        # 交叉注意力层
        cross_attn_keys = [
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_token_to_image.q_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_token_to_image.k_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_token_to_image.v_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_token_to_image.out_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_image_to_token.q_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_image_to_token.k_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_image_to_token.v_proj.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.cross_attn_image_to_token.out_proj.weight"
        ]
        
        for key in cross_attn_keys:
            weight = torch.empty(transformer_dim, transformer_dim)
            nn.init.xavier_uniform_(weight)
            target_state_dict[key] = weight
        
        # MLP层
        mlp_keys = [
            f"mask_decoder.transformer.layers.{layer_idx}.mlp.lin1.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.mlp.lin2.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.mlp.lin1.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.mlp.lin2.bias"
        ]
        
        for key in mlp_keys:
            if "lin1" in key and "weight" in key:
                weight = torch.empty(transformer_dim * 8, transformer_dim)  # 8x expansion
                nn.init.xavier_uniform_(weight)
                target_state_dict[key] = weight
            elif "lin2" in key and "weight" in key:
                weight = torch.empty(transformer_dim, transformer_dim * 8)
                nn.init.xavier_uniform_(weight)
                target_state_dict[key] = weight
            elif "lin1" in key and "bias" in key:
                bias = torch.zeros(transformer_dim * 8)
                target_state_dict[key] = bias
            elif "lin2" in key and "bias" in key:
                bias = torch.zeros(transformer_dim)
                target_state_dict[key] = bias
        
        # LayerNorm层
        norm_keys = [
            f"mask_decoder.transformer.layers.{layer_idx}.norm1.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.norm2.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.norm3.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.norm4.weight",
            f"mask_decoder.transformer.layers.{layer_idx}.norm1.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.norm2.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.norm3.bias",
            f"mask_decoder.transformer.layers.{layer_idx}.norm4.bias"
        ]
        
        for key in norm_keys:
            if "weight" in key:
                weight = torch.ones(transformer_dim)
                target_state_dict[key] = weight
            else:  # bias
                bias = torch.zeros(transformer_dim)
                target_state_dict[key] = bias
    
    def _validate_migrated_weights(self, target_state_dict: Dict[str, torch.Tensor]):
        """验证迁移后的权重"""
        self.logger.info("验证迁移后的权重...")
        
        validation_errors = []
        validation_warnings = []
        
        # 检查权重形状
        for key, weight in target_state_dict.items():
            if torch.isnan(weight).any():
                validation_errors.append(f"权重 {key} 包含NaN值")
            
            if torch.isinf(weight).any():
                validation_errors.append(f"权重 {key} 包含无穷值")
            
            # 检查权重范围
            weight_std = weight.std().item()
            if weight_std > 10.0:
                validation_warnings.append(f"权重 {key} 标准差过大: {weight_std:.4f}")
            elif weight_std < 1e-6:
                validation_warnings.append(f"权重 {key} 标准差过小: {weight_std:.4f}")
        
        # 报告验证结果
        if validation_errors:
            error_msg = "权重验证失败:\n" + "\n".join(f"- {e}" for e in validation_errors)
            raise ValueError(error_msg)
        
        if validation_warnings and self.verbose:
            warning_msg = "权重验证警告:\n" + "\n".join(f"- {w}" for w in validation_warnings)
            self.logger.warning(warning_msg)
        
        self.logger.info("权重验证通过")
    
    def get_migration_report(self) -> Dict[str, Any]:
        """获取迁移报告"""
        return {
            "source_config": {
                "model_size": self.source_config.model_size,
                "depth": self.source_config.depth,
                "embed_dim": self.source_config.embed_dim,
                "moe_num_experts": self.source_config.moe_num_experts,
                "estimated_params": self.source_config.estimate_parameters()
            },
            "target_config": {
                "model_size": self.target_config.model_size,
                "depth": self.target_config.depth,
                "embed_dim": self.target_config.embed_dim,
                "moe_num_experts": self.target_config.moe_num_experts,
                "estimated_params": self.target_config.estimate_parameters()
            },
            "migration_stats": self.migration_stats,
            "migration_strategy": self.migration_strategy,
            "expert_initialization": self.expert_initialization,
            "dimension_interpolation": self.dimension_interpolation,
            "validation_enabled": self.validation_enabled
        }


def create_weight_migration_manager(
    source_model_size: str,
    target_model_size: str,
    **kwargs
) -> WeightMigrationManager:
    """
    创建权重迁移管理器的便捷函数
    
    Args:
        source_model_size: 源模型规模
        target_model_size: 目标模型规模
        **kwargs: 其他参数
        
    Returns:
        WeightMigrationManager实例
    """
    source_config = CWSAMScalingConfig(source_model_size)
    target_config = CWSAMScalingConfig(target_model_size)
    
    return WeightMigrationManager(
        source_config=source_config,
        target_config=target_config,
        **kwargs
    )


if __name__ == "__main__":
    # 测试权重迁移管理器
    print("测试权重迁移管理器")
    print("=" * 50)
    
    # 测试不同的迁移场景
    test_cases = [
        ("1.5B", "2B"),
        ("2B", "3B"),
        ("3B", "5B"),
        ("5B", "7B")
    ]
    
    for source_size, target_size in test_cases:
        try:
            print(f"\n测试 {source_size} -> {target_size} 迁移...")
            
            migration_manager = create_weight_migration_manager(
                source_size, target_size, verbose=False
            )
            
            # 创建模拟的源权重
            source_state_dict = {
                "patch_embed.proj.weight": torch.randn(1280, 3, 16, 16),
                "pos_embed": torch.randn(1, 64, 64, 1280),
                "blocks.0.norm1.weight": torch.randn(1280),
                "blocks.0.attn.qkv.weight": torch.randn(3840, 1280),
            }
            
            # 执行迁移
            target_state_dict = migration_manager.migrate_weights(source_state_dict)
            
            print(f"✓ 迁移成功，目标权重数量: {len(target_state_dict)}")
            
            # 获取迁移报告
            report = migration_manager.get_migration_report()
            print(f"参数增长: {report['migration_stats']['param_increase_ratio']:.2f}x")
            
        except Exception as e:
            print(f"✗ 迁移失败: {e}")