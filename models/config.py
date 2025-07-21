"""
CWSAM模型参数扩展配置管理系统

该模块提供了CWSAM模型从1.5B到10B参数量扩展的配置管理功能，
包括预定义配置、参数量估算和配置验证。
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# 避免在测试时导入torch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class ModelScalingConfig:
    """模型扩展配置数据类"""
    model_size: str
    moe_num_experts: int
    depth: int
    embed_dim: int
    num_heads: int
    moe_start_layer: int
    mlp_ratio: float = 4.0
    use_multi_scale: bool = False
    decoder_depth: int = 2
    patch_size: int = 16
    img_size: int = 1024
    
    def __post_init__(self):
        """初始化后验证配置"""
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(f"嵌入维度 {self.embed_dim} 必须能被注意力头数 {self.num_heads} 整除")
        
        if self.moe_start_layer >= self.depth:
            raise ValueError(f"MoE开始层 {self.moe_start_layer} 必须小于总深度 {self.depth}")
    
    def estimate_parameters(self) -> int:
        """估算模型参数量 - 基于实际的CWSAM架构"""
        
        # 基础参数计算 - 基于ViT-Huge的基础架构
        base_vit_params = 632_000_000  # ViT-Huge基础参数量
        
        # 根据嵌入维度调整基础参数
        embed_scale = (self.embed_dim / 1280) ** 2  # 相对于标准1280维度
        depth_scale = self.depth / 32  # 相对于标准32层
        
        # 调整后的基础参数
        adjusted_base_params = int(base_vit_params * embed_scale * depth_scale)
        
        # MoE额外参数计算 - 更保守的估算
        moe_layers = max(0, self.depth - self.moe_start_layer)
        if moe_layers > 0:
            # 每个MoE层相对于标准FFN的额外参数
            mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
            standard_ffn_params = 2 * self.embed_dim * mlp_hidden_dim
            
            # MoE的有效参数增长 - 考虑到稀疏激活
            # 实际上只有top-k专家被激活，所以有效参数增长有限
            k = 4  # top-k路由
            effective_expert_ratio = min(self.moe_num_experts / 16, 8)  # 相对于基础16专家的倍数，最大8倍
            
            moe_extra_params = moe_layers * standard_ffn_params * (effective_expert_ratio - 1)
            
            # 门控网络参数 (相对较小)
            gate_params = moe_layers * self.embed_dim * self.moe_num_experts
            
            total_moe_params = int(moe_extra_params + gate_params)
        else:
            total_moe_params = 0
        
        # 解码器参数 (相对固定)
        decoder_params = self._estimate_decoder_params()
        
        # 多尺度特征模块参数
        multi_scale_params = 0
        if self.use_multi_scale:
            multi_scale_params = self._estimate_multi_scale_params()
        
        # 总参数量
        total_params = adjusted_base_params + total_moe_params + decoder_params + multi_scale_params
        
        return total_params
    
    def _estimate_decoder_params(self) -> int:
        """估算解码器参数量"""
        transformer_dim = 256  # 默认解码器transformer维度
        
        # TwoWayTransformer参数
        transformer_params = self.decoder_depth * (
            # 自注意力
            4 * transformer_dim * transformer_dim +
            # 交叉注意力 (两个方向)
            2 * 4 * transformer_dim * transformer_dim +
            # MLP
            2 * transformer_dim * (transformer_dim * 4) +
            # LayerNorm
            6 * transformer_dim * 2  # 每层3个LayerNorm
        )
        
        # 输出头参数
        output_head_params = (
            transformer_dim * 3 +  # 3个mask输出
            transformer_dim * 1    # IoU预测头
        )
        
        return transformer_params + output_head_params
    
    def _estimate_multi_scale_params(self) -> int:
        """估算多尺度特征模块参数量"""
        scales = [1, 2, 4]  # 默认尺度
        
        # 尺度卷积参数
        scale_conv_params = len(scales) * (
            self.embed_dim * self.embed_dim * 9  # 3x3卷积
        )
        
        # 融合卷积参数
        fusion_conv_params = (
            len(scales) * self.embed_dim * self.embed_dim  # 1x1卷积
        )
        
        # 上采样卷积参数
        upsample_params = (len(scales) - 1) * (
            self.embed_dim * self.embed_dim * 16  # 转置卷积
        )
        
        return scale_conv_params + fusion_conv_params + upsample_params
    
    def estimate_memory_usage(self, batch_size: int = 1, precision: str = "fp32") -> Dict[str, float]:
        """估算内存使用量 (GB)"""
        bytes_per_param = 4 if precision == "fp32" else 2  # fp16
        
        # 模型参数内存
        model_memory = self.estimate_parameters() * bytes_per_param / (1024**3)
        
        # 激活内存估算 (粗略估计)
        sequence_length = (self.img_size // self.patch_size) ** 2
        activation_memory = (
            batch_size * sequence_length * self.embed_dim * self.depth * 
            bytes_per_param / (1024**3)
        )
        
        # 梯度内存 (训练时)
        gradient_memory = model_memory
        
        # 优化器状态内存 (Adam)
        optimizer_memory = model_memory * 2  # momentum + variance
        
        return {
            "model": model_memory,
            "activation": activation_memory,
            "gradient": gradient_memory,
            "optimizer": optimizer_memory,
            "total_training": model_memory + activation_memory + gradient_memory + optimizer_memory,
            "total_inference": model_memory + activation_memory
        }


class CWSAMScalingConfig:
    """CWSAM模型扩展配置管理器"""
    
    # 预定义的模型配置
    PREDEFINED_CONFIGS = {
        "1.5B": {
            "moe_num_experts": 16,
            "depth": 32,
            "embed_dim": 1280,
            "num_heads": 16,
            "moe_start_layer": 28,
            "use_multi_scale": False,
            "decoder_depth": 2
        },
        "2B": {
            "moe_num_experts": 32,
            "depth": 36,
            "embed_dim": 1280,
            "num_heads": 16,
            "moe_start_layer": 24,
            "use_multi_scale": False,
            "decoder_depth": 3
        },
        "3B": {
            "moe_num_experts": 64,
            "depth": 40,
            "embed_dim": 1536,
            "num_heads": 24,
            "moe_start_layer": 20,
            "use_multi_scale": True,
            "decoder_depth": 4
        },
        "5B": {
            "moe_num_experts": 128,
            "depth": 48,
            "embed_dim": 2048,
            "num_heads": 32,
            "moe_start_layer": 16,
            "use_multi_scale": True,
            "decoder_depth": 6
        },
        "7B": {
            "moe_num_experts": 256,
            "depth": 56,
            "embed_dim": 2560,
            "num_heads": 40,
            "moe_start_layer": 12,
            "use_multi_scale": True,
            "decoder_depth": 8
        },
        "10B": {
            "moe_num_experts": 512,
            "depth": 64,
            "embed_dim": 3072,
            "num_heads": 48,
            "moe_start_layer": 8,
            "use_multi_scale": True,
            "decoder_depth": 12
        }
    }
    
    def __init__(self, model_size: str):
        """
        初始化配置管理器
        
        Args:
            model_size: 模型规模 ("1.5B", "2B", "3B", "5B", "7B", "10B")
        """
        if model_size not in self.PREDEFINED_CONFIGS:
            raise ValueError(f"不支持的模型规模: {model_size}. 支持的规模: {list(self.PREDEFINED_CONFIGS.keys())}")
        
        self.model_size = model_size
        self.config = self._create_config(model_size)
    
    def _create_config(self, model_size: str) -> ModelScalingConfig:
        """创建模型配置"""
        base_config = self.PREDEFINED_CONFIGS[model_size].copy()
        return ModelScalingConfig(
            model_size=model_size,
            **base_config
        )
    
    def get_config(self) -> ModelScalingConfig:
        """获取配置对象"""
        return self.config
    
    def get_config_dict(self) -> Dict:
        """获取配置字典"""
        return {
            "model_size": self.config.model_size,
            "moe_num_experts": self.config.moe_num_experts,
            "depth": self.config.depth,
            "embed_dim": self.config.embed_dim,
            "num_heads": self.config.num_heads,
            "moe_start_layer": self.config.moe_start_layer,
            "mlp_ratio": self.config.mlp_ratio,
            "use_multi_scale": self.config.use_multi_scale,
            "decoder_depth": self.config.decoder_depth,
            "patch_size": self.config.patch_size,
            "img_size": self.config.img_size
        }
    
    def estimate_parameters(self) -> int:
        """估算模型参数量"""
        return self.config.estimate_parameters()
    
    def estimate_memory_usage(self, batch_size: int = 1, precision: str = "fp32") -> Dict[str, float]:
        """估算内存使用量"""
        return self.config.estimate_memory_usage(batch_size, precision)
    
    def print_config_summary(self):
        """打印配置摘要"""
        config = self.config
        params = self.estimate_parameters()
        memory = self.estimate_memory_usage()
        
        print(f"\n=== CWSAM {self.model_size} 配置摘要 ===")
        print(f"模型规模: {self.model_size}")
        print(f"预估参数量: {params:,} ({params/1e9:.2f}B)")
        print(f"MoE专家数量: {config.moe_num_experts}")
        print(f"模型深度: {config.depth}")
        print(f"嵌入维度: {config.embed_dim}")
        print(f"注意力头数: {config.num_heads}")
        print(f"MoE开始层: {config.moe_start_layer}")
        print(f"多尺度特征: {'启用' if config.use_multi_scale else '禁用'}")
        print(f"解码器深度: {config.decoder_depth}")
        print(f"\n内存使用估算 (FP32):")
        print(f"  推理内存: {memory['total_inference']:.2f} GB")
        print(f"  训练内存: {memory['total_training']:.2f} GB")
        print("=" * 40)
    
    @classmethod
    def list_available_sizes(cls) -> List[str]:
        """列出所有可用的模型规模"""
        return list(cls.PREDEFINED_CONFIGS.keys())
    
    @classmethod
    def create_custom_config(
        cls,
        model_size: str,
        moe_num_experts: int,
        depth: int,
        embed_dim: int,
        num_heads: int,
        moe_start_layer: int,
        **kwargs
    ) -> 'CWSAMScalingConfig':
        """创建自定义配置"""
        # 创建临时配置管理器
        temp_config = cls("1.5B")  # 使用默认配置作为基础
        
        # 更新配置
        custom_config = ModelScalingConfig(
            model_size=model_size,
            moe_num_experts=moe_num_experts,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            moe_start_layer=moe_start_layer,
            **kwargs
        )
        
        temp_config.config = custom_config
        temp_config.model_size = model_size
        
        return temp_config


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_scaling_config(config: ModelScalingConfig) -> List[str]:
        """
        验证扩展配置的合理性
        
        Args:
            config: 模型配置对象
            
        Returns:
            错误信息列表，空列表表示验证通过
        """
        errors = []
        
        # 1. 检查专家数量
        if config.moe_num_experts < 8:
            errors.append("MoE专家数量不应少于8个")
        elif config.moe_num_experts > 512:
            errors.append("MoE专家数量不应超过512个")
        elif not ConfigValidator._is_power_of_2(config.moe_num_experts):
            errors.append("MoE专家数量建议为2的幂次方以优化计算效率")
        
        # 2. 检查嵌入维度和注意力头数的关系
        if config.embed_dim % config.num_heads != 0:
            errors.append("嵌入维度必须能被注意力头数整除")
        
        head_dim = config.embed_dim // config.num_heads
        if head_dim < 32 or head_dim > 128:
            errors.append(f"每个注意力头的维度 ({head_dim}) 建议在32-128之间")
        
        # 3. 检查模型深度
        if config.depth < 12:
            errors.append("模型深度不应少于12层")
        elif config.depth > 64:
            errors.append("模型深度不应超过64层")
        
        # 4. 检查MoE开始层
        if config.moe_start_layer >= config.depth:
            errors.append("MoE开始层必须小于总深度")
        elif config.moe_start_layer < 0:
            errors.append("MoE开始层不能为负数")
        
        # 5. 检查嵌入维度
        if config.embed_dim < 512:
            errors.append("嵌入维度不应小于512")
        elif config.embed_dim > 4096:
            errors.append("嵌入维度不应超过4096")
        elif config.embed_dim % 64 != 0:
            errors.append("嵌入维度建议为64的倍数以优化计算效率")
        
        # 6. 检查内存需求
        try:
            memory_usage = config.estimate_memory_usage(batch_size=1, precision="fp32")
            training_memory = memory_usage["total_training"]
            
            # 根据模型规模设置不同的内存限制
            memory_limits = {
                "1.5B": 24,  # 24GB
                "2B": 32,    # 32GB
                "3B": 48,    # 48GB
                "5B": 80,    # 80GB
                "7B": 120,   # 120GB
                "10B": 160   # 160GB
            }
            
            limit = memory_limits.get(config.model_size, 80)
            if training_memory > limit:
                errors.append(f"预估训练内存需求 {training_memory:.1f}GB 超过建议限制 {limit}GB")
        
        except Exception as e:
            errors.append(f"内存估算失败: {str(e)}")
        
        # 7. 检查10B模型的特殊约束
        if config.model_size == "10B":
            if config.moe_num_experts > 512:
                errors.append("10B模型专家数量不应超过512")
            if config.embed_dim > 3072:
                errors.append("10B模型嵌入维度不应超过3072")
            if not config.use_multi_scale:
                errors.append("10B模型建议启用多尺度特征以提升性能")
        
        # 8. 检查解码器深度
        if config.decoder_depth < 2:
            errors.append("解码器深度不应少于2层")
        elif config.decoder_depth > 12:
            errors.append("解码器深度不应超过12层")
        
        return errors
    
    @staticmethod
    def _is_power_of_2(n: int) -> bool:
        """检查是否为2的幂次方"""
        return n > 0 and (n & (n - 1)) == 0
    
    @staticmethod
    def validate_hardware_compatibility(config: ModelScalingConfig, available_gpus: int = 1, gpu_memory_gb: int = 24) -> List[str]:
        """
        验证硬件兼容性
        
        Args:
            config: 模型配置
            available_gpus: 可用GPU数量
            gpu_memory_gb: 单个GPU内存大小(GB)
            
        Returns:
            警告信息列表
        """
        warnings = []
        
        memory_usage = config.estimate_memory_usage(batch_size=1, precision="fp32")
        total_gpu_memory = available_gpus * gpu_memory_gb
        
        # 检查推理内存
        if memory_usage["total_inference"] > total_gpu_memory:
            warnings.append(f"推理内存需求 {memory_usage['total_inference']:.1f}GB 超过可用GPU内存 {total_gpu_memory}GB")
        
        # 检查训练内存
        if memory_usage["total_training"] > total_gpu_memory:
            warnings.append(f"训练内存需求 {memory_usage['total_training']:.1f}GB 超过可用GPU内存 {total_gpu_memory}GB")
            warnings.append("建议使用混合精度训练(FP16)或梯度检查点技术")
        
        # 针对大模型的建议
        if config.model_size in ["7B", "10B"] and available_gpus < 4:
            warnings.append(f"{config.model_size}模型建议使用至少4个GPU进行训练")
        
        if config.model_size == "10B" and available_gpus < 8:
            warnings.append("10B模型建议使用至少8个GPU以获得最佳性能")
        
        return warnings


# 便捷函数
def create_config(model_size: str) -> CWSAMScalingConfig:
    """创建配置管理器的便捷函数"""
    return CWSAMScalingConfig(model_size)


def validate_config(config: ModelScalingConfig) -> Tuple[bool, List[str]]:
    """验证配置的便捷函数"""
    errors = ConfigValidator.validate_scaling_config(config)
    return len(errors) == 0, errors


def list_model_sizes() -> List[str]:
    """列出所有支持的模型规模"""
    return CWSAMScalingConfig.list_available_sizes()


if __name__ == "__main__":
    # 示例用法
    print("CWSAM模型配置管理系统示例")
    print("=" * 50)
    
    # 创建不同规模的配置
    for size in ["1.5B", "3B", "5B", "10B"]:
        try:
            config_manager = create_config(size)
            config_manager.print_config_summary()
            
            # 验证配置
            is_valid, errors = validate_config(config_manager.get_config())
            if not is_valid:
                print(f"配置验证失败: {errors}")
            else:
                print("✓ 配置验证通过")
            
            print()
        except Exception as e:
            print(f"创建 {size} 配置失败: {e}")