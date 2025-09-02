import argparse
import os
import yaml
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm

import models
from test import de_normalize, onehot_to_mask

def prepare_image(image_path, input_size=512):
    """准备输入图像"""
    # 使用 PIL.Image.BILINEAR 进行更快的缩放
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.430, 0.411, 0.296],
                            std=[0.213, 0.156, 0.143])
    ])
    
    # 读取并处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加batch维度

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                       default='/mnt/fanfq/project/code/cwsam_10b/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', 
                       default='/mnt/fanfq/project/code/cwsam_10b/save/dinov3-vit7b-0830-1653/model_epoch_20.pth',
                       help='模型权重路径')
    parser.add_argument('--input_dir', 
                       default='/mnt/fanfq/data/fan/data/XinTong/05m-2m/patch_1024',
                       help='输入图像文件夹')
    parser.add_argument('--output_dir', 
                       default='/mnt/fanfq/data/fan/data/XinTong/05m-2m/dinov3—20',
                       help='输出结果文件夹')
    parser.add_argument('--device', 
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='选择推理设备 (cuda/cpu)')
    parser.add_argument('--task_id', 
                       type=int,
                       default=0,
                       help='指定推理任务ID (0: XinTong, 1: UAV等)')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 获取任务配置信息
    task_configs = config.get('task_configs', [])
    
    # 如果顶级没有task_configs，尝试从model.args中获取
    if not task_configs and 'model' in config and 'args' in config['model']:
        task_configs = config['model']['args'].get('task_configs', [])
    
    current_task_config = None
    
    # 查找当前任务ID对应的配置
    for task_config in task_configs:
        if task_config['task_id'] == args.task_id:
            current_task_config = task_config
            break
    
    if current_task_config is None:
        # 如果没有找到多任务配置，尝试使用单任务配置
        if 'test_dataset' in config and 'dataset' in config['test_dataset'] and 'args' in config['test_dataset']['dataset']:
            color_palette = config['test_dataset']['dataset']['args']['palette']
            num_classes = len(color_palette)
            task_name = "单任务"
        elif args.task_id == 0:
            # 为单任务模型提供默认配置
            print(f"警告: 没有找到任务配置，使用默认单任务配置 (task_id={args.task_id})")
            color_palette = [[i, i, i] for i in range(256)]  # 默认调色板
            num_classes = 256
            task_name = "默认单任务"
        else:
            raise ValueError(f"无法找到任务ID {args.task_id} 对应的配置，请检查配置文件或使用正确的task_id")
    else:
        # 使用多任务配置
        color_palette = current_task_config['palete']
        num_classes = current_task_config['num_classes']
        task_name = current_task_config['name']
    
    print(f"使用任务: {task_name} (ID: {args.task_id}), 类别数: {num_classes}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() and 'cuda' in args.device else 'cpu')
    
    # 设置模型为channels_last格式以提高性能
    with torch.no_grad():
        model = models.make(config['model']).to(device)
        if device.type == 'cuda':
            model = model.to(memory_format=torch.channels_last)
        
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    
    # 处理不同格式的checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 训练时保存的checkpoint格式
        model_state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'model' in checkpoint:
        # 从checkpoint['model']中提取权重
        model_state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # 常见命名：'state_dict'
        model_state_dict = checkpoint['state_dict']
    else:
        # 直接保存的模型状态字典
        model_state_dict = checkpoint

    # 常见DDP/Lightning前缀清理
    if isinstance(model_state_dict, dict):
        if any(k.startswith('module.') for k in model_state_dict.keys()):
            model_state_dict = {k[len('module.'):]: v for k, v in model_state_dict.items()}
        if any(k.startswith('model.') for k in model_state_dict.keys()):
            model_state_dict = {k[len('model.'):]: v for k, v in model_state_dict.items()}
    
    # 加载模型权重，使用strict=False来忽略不匹配的键
    missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=False)
    
    if missing_keys:
        print(f"警告: 模型中缺少以下权重键: {missing_keys}")
    if unexpected_keys:
        print(f"警告: checkpoint中包含以下未预期的权重键: {unexpected_keys}")
    
    model.eval()
    
    # 如果支持，使用torch.compile加速推理（PyTorch 2.0+）
    if hasattr(torch, 'compile') and device.type == 'cuda':
        try:
            print("使用 torch.compile 优化模型...")
            model = torch.compile(model, mode='max-autotune')
            print("torch.compile 优化完成")
        except Exception as e:
            print(f"torch.compile 优化失败，使用原始模型: {e}")
    
    # 预热模型（重要：让模型完成初始化和优化）
    if device.type == 'cuda':
        print("预热模型...")
        dummy_input = torch.randn(1, 3, 512, 512).to(device, memory_format=torch.channels_last)
        with torch.no_grad():
            for _ in range(3):  # 预热几次
                _ = model.infer(dummy_input)
        print("模型预热完成")
        del dummy_input
        torch.cuda.empty_cache()

    # 获取所有图像文件
    image_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 仅在使用CUDA时启用优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        # 添加训练脚本中的优化设置
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
            try:
                from torch.backends.cuda import sdp_kernel
                sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
            except Exception:
                pass
        except Exception:
            pass
    
    # 根据设备类型调整批处理大小 - 针对大模型使用更大批次
    if device.type == 'cuda':
        # 检查GPU内存来动态调整批处理大小
        total_memory = torch.cuda.get_device_properties(device.index).total_memory
        if total_memory > 90 * 1024**3:  # 大于90GB（H20）
            BATCH_SIZE = 24
        elif total_memory > 40 * 1024**3:  # 大于40GB
            BATCH_SIZE = 16
        else:
            BATCH_SIZE = 8
        print(f"检测到GPU内存: {total_memory / 1024**3:.1f}GB, 使用批处理大小: {BATCH_SIZE}")
    else:
        BATCH_SIZE = 2
    
    # 处理每张图像
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="处理图像批次"):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_inputs = []
        
        # 准备批处理输入 - 优化数据加载
        for image_file in batch_files:
            image_path = os.path.join(args.input_dir, image_file)
            input_tensor = prepare_image(image_path)
            batch_inputs.append(input_tensor)
            
        batch_inputs = torch.cat(batch_inputs, dim=0).to(device, non_blocking=True)
        # 转换为channels_last格式以提高性能
        if device.type == 'cuda':
            batch_inputs = batch_inputs.to(memory_format=torch.channels_last)
        
        # 批量推理
        with torch.no_grad():
            # 使用混合精度推理加速
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                # 检查模型是否支持多任务
                if hasattr(model, 'task_configs') and hasattr(model, 'mask_decoders'):
                    # 多任务模型，传入task_id
                    batch_outputs = model.infer(batch_inputs, task_id=args.task_id)
                else:
                    # 单任务模型
                    batch_outputs = model.infer(batch_inputs)
                
                # 获取预测概率
                batch_probs = torch.softmax(batch_outputs, dim=1)
            
        # 处理每个输出
        for idx, image_file in enumerate(batch_files):
            output_masks = batch_outputs[idx].cpu()
            # confidence_map = confidence[idx].cpu().numpy()
            
            # 转换为彩色掩码
            binary_mask = onehot_to_mask(output_masks, palette=color_palette)
            
            # 保存掩码
            mask_filename = f'{os.path.splitext(image_file)[0]}_task{args.task_id}_mask.png'
            mask_path = os.path.join(args.output_dir, 'masks', mask_filename)
            Image.fromarray(np.uint8(binary_mask)).convert('RGB').save(mask_path)
            
            # # 保存置信度图
            # confidence_path = os.path.join(args.output_dir, 'masks', f'{os.path.splitext(image_file)[0]}_confidence.npy')
            # np.save(confidence_path, confidence_map)

if __name__ == '__main__':
    main()
