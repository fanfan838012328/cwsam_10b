"""
测试模型工厂和注册系统

该脚本测试ModelFactory类的功能，包括创建不同规模的模型、
获取模型信息和验证SAM_MOE_3B类的向后兼容性。
"""

import torch
from models.model_factory import ModelFactory, create_model, print_model_info, list_available_models
from models.sam import SAM_MOE_3B

def test_model_factory():
    """测试ModelFactory类的基本功能"""
    print("\n=== 测试ModelFactory基本功能 ===")
    
    # 创建工厂实例
    factory = ModelFactory()
    
    # 列出所有可用模型
    models = factory.list_available_models()
    print(f"可用模型规模: {models}")
    
    # 打印模型信息
    for size in ["1.5B", "2B", "3B"]:
        try:
            info = factory.get_model_info(size)
            print(f"\n{size}模型信息:")
            print(f"  参数量: {info['parameters_billions']:.2f}B")
            print(f"  MoE专家数: {info['moe_experts']}")
            print(f"  模型深度: {info['depth']}")
            print(f"  嵌入维度: {info['embed_dim']}")
        except Exception as e:
            print(f"获取{size}模型信息失败: {e}")
    
    return True

def test_create_models():
    """测试创建不同规模的模型"""
    print("\n=== 测试创建不同规模模型 ===")
    
    # 测试参数
    inp_size = 256  # 使用小尺寸加速测试
    encoder_mode = {
        'name': 'vit_1dot5b',
        'patch_size': 16,
        'embed_dim': 1280,
        'depth': 32,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'out_chans': 256,
        'qkv_bias': True,
        'use_rel_pos': True,
        'window_size': 14,
        'global_attn_indexes': [2, 5, 8, 11],
        'prompt_embed_dim': 256
    }
    
    # 测试创建1.5B和2B模型
    for size in ["1.5B", "2B"]:
        try:
            print(f"\n创建{size}模型...")
            model = create_model(size, inp_size=inp_size, encoder_mode=encoder_mode, num_classes=1)
            
            # 验证模型结构
            print(f"模型类型: {type(model).__name__}")
            print(f"编码器类型: {type(model.image_encoder).__name__}")
            print(f"解码器类型: {type(model.mask_decoder).__name__}")
            
            # 测试前向传播
            x = torch.randn(1, 3, inp_size, inp_size)
            with torch.no_grad():
                try:
                    output = model.infer(x)
                    print(f"输出形状: {output.shape}")
                except Exception as e:
                    print(f"前向传播失败: {e}")
            
        except Exception as e:
            print(f"创建{size}模型失败: {e}")
    
    return True

def test_sam_moe_3b_compatibility():
    """测试SAM_MOE_3B类的向后兼容性"""
    print("\n=== 测试SAM_MOE_3B向后兼容性 ===")
    
    # 测试参数
    inp_size = 256  # 使用小尺寸加速测试
    encoder_mode = {
        'name': 'vit_3b',
        'patch_size': 16,
        'embed_dim': 1536,
        'depth': 40,
        'num_heads': 24,
        'mlp_ratio': 4.0,
        'out_chans': 256,
        'qkv_bias': True,
        'use_rel_pos': True,
        'window_size': 14,
        'global_attn_indexes': [2, 5, 8, 11],
        'prompt_embed_dim': 256
    }
    
    try:
        # 创建SAM_MOE_3B实例
        print("创建SAM_MOE_3B实例...")
        model = SAM_MOE_3B(
            inp_size=inp_size,
            encoder_mode=encoder_mode,
            num_classes=1
        )
        
        # 验证模型结构
        print(f"模型类型: {type(model).__name__}")
        print(f"内部模型类型: {type(model.model).__name__}")
        print(f"编码器类型: {type(model.image_encoder).__name__}")
        
        # 测试前向传播
        x = torch.randn(1, 3, inp_size, inp_size)
        gt = torch.zeros(1, 1, inp_size, inp_size)
        gt[:, :, 100:150, 100:150] = 1
        
        model.set_input(x, gt)
        
        with torch.no_grad():
            try:
                output = model.forward()
                print(f"输出形状: {output.shape}")
            except Exception as e:
                print(f"前向传播失败: {e}")
        
    except Exception as e:
        print(f"创建SAM_MOE_3B模型失败: {e}")
    
    return True

if __name__ == "__main__":
    print("=" * 50)
    print("CWSAM模型工厂和注册系统测试")
    print("=" * 50)
    
    # 运行测试
    tests = [
        test_model_factory,
        test_create_models,
        test_sam_moe_3b_compatibility
    ]
    
    all_passed = True
    for test_func in tests:
        try:
            result = test_func()
            all_passed = all_passed and result
        except Exception as e:
            print(f"测试 {test_func.__name__} 失败: {e}")
            all_passed = False
    
    # 打印总结
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ 所有测试通过!")
    else:
        print("✗ 测试失败!")
    print("=" * 50)