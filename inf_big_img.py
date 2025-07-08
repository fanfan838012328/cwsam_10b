import argparse
import os
import yaml
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

import models
from test import de_normalize, onehot_to_mask

def prepare_image(image_path, input_size=1024, original_resolution=True):
    """准备输入图像
    
    Args:
        image_path: 图像路径
        input_size: 模型输入尺寸
        original_resolution: 是否保留原始分辨率信息
    """
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # 保存原始尺寸 (W, H)
    
    # 创建转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 处理图像
    if original_resolution:
        # 保持原始纵横比的情况下缩放
        image_tensor = transform(image)
        # 添加batch维度
        image_tensor = image_tensor.unsqueeze(0)
        
        # 获取原始尺寸(H, W)供后续处理使用
        original_h, original_w = image_tensor.shape[-2], image_tensor.shape[-1]
        return image_tensor, (original_h, original_w)
    else:
        # 使用固定大小调整
        resize_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        image_tensor = resize_transform(image)
        return image_tensor.unsqueeze(0), (input_size, input_size)  # 添加batch维度

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                       default='/mnt/fanfq/data/fan/cwsam/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', 
                       default='/mnt/fanfq/data/fan/cwsam/save/XinTong_sam_vit_h_moe_3b/model_epoch_121.pth',
                       help='模型权重路径')
    parser.add_argument('--input_dir', 
                       default='/mnt/fanfq/project/data/dronetest/DroneTest',
                       help='输入图像文件夹')
    parser.add_argument('--output_dir', 
                       default='/mnt/fanfq/project/data/dronetest/DroneTest_output',
                       help='输出结果文件夹')
    parser.add_argument('--device', 
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='选择推理设备 (cuda/cpu)')
    parser.add_argument('--keep_ratio', action='store_true',
                       help='是否保持图像原始宽高比')
    parser.add_argument('--tile_size', type=int, default=1024,
                       help='切片尺寸，适用于超大分辨率图像')
    parser.add_argument('--tile_overlap', type=int, default=256,
                       help='切片重叠部分的大小')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 获取调色板和类别数
    color_palette = config['test_dataset']['dataset']['args']['palette']
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'masks'), exist_ok=True)

    # 加载模型
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    model = models.make(config['model']).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # 获取所有图像文件
    image_files = [f for f in os.listdir(args.input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # 仅在使用CUDA时启用优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    
    # 根据设备类型调整批处理大小
    BATCH_SIZE = 4 if device.type == 'cuda' else 1  # 对于高分辨率图像，降低批处理大小防止内存溢出
    
    # 处理每张图像
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="处理图像批次"):
        batch_files = image_files[i:i + BATCH_SIZE]
        
        for image_file in batch_files:
            image_path = os.path.join(args.input_dir, image_file)
            
            # 判断图像尺寸是否需要采用切片处理
            img = Image.open(image_path)
            width, height = img.size
            
            if width > args.tile_size or height > args.tile_size:
                # 对大尺寸图像进行切片处理
                final_mask = process_large_image(model, image_path, device, args.tile_size, args.tile_overlap)
            else:
                # 对小尺寸图像直接处理
                input_tensor, original_size = prepare_image(image_path, input_size=1024, original_resolution=args.keep_ratio)
                input_tensor = input_tensor.to(device)
                
                # 推理
                with torch.no_grad():
                    output = model.infer(input_tensor)
                    
                    # 如果模型不直接支持自定义大小输出，手动调整输出大小
                    if args.keep_ratio and hasattr(model, 'postprocess_masks'):
                        # 使用原始尺寸信息
                        final_mask = output
                    else:
                        # 手动调整输出到原始尺寸
                        final_mask = F.interpolate(output, size=(height, width), mode='bilinear', align_corners=False)
            
            # 获取预测概率
            probs = torch.softmax(final_mask, dim=1)
            # 获取最大概率值
            confidence, _ = torch.max(probs, dim=1)
            
            # 转换为彩色掩码
            binary_mask = onehot_to_mask(final_mask[0].cpu(), palette=color_palette)
            
            # 保存掩码
            mask_path = os.path.join(args.output_dir, 'masks', f'{os.path.splitext(image_file)[0]}_mask.png')
            Image.fromarray(np.uint8(binary_mask)).convert('RGB').save(mask_path)
            
            # 可选：保存置信度图
            # confidence_path = os.path.join(args.output_dir, 'masks', f'{os.path.splitext(image_file)[0]}_confidence.npy')
            # np.save(confidence_path, confidence[0].cpu().numpy())

def process_large_image(model, image_path, device, tile_size=1024, overlap=256):
    """处理大尺寸图像的切片推理方法"""
    # 读取原始图像
    img = Image.open(image_path).convert('RGB')
    width, height = img.size
    
    # 归一化转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 初始化最终输出掩码和置信度图
    final_mask = None
    confidence_map = None
    
    # 计算切片的行列数
    rows = max(1, int(np.ceil((height - overlap) / (tile_size - overlap))))
    cols = max(1, int(np.ceil((width - overlap) / (tile_size - overlap))))
    
    for r in range(rows):
        for c in range(cols):
            # 计算当前切片的坐标
            y1 = min(r * (tile_size - overlap), height - tile_size)
            x1 = min(c * (tile_size - overlap), width - tile_size)
            y2 = min(y1 + tile_size, height)
            x2 = min(x1 + tile_size, width)
            
            # 确保最后一行/列的切片不小于tile_size
            if y2 - y1 < tile_size and height >= tile_size:
                y1 = height - tile_size
            if x2 - x1 < tile_size and width >= tile_size:
                x1 = width - tile_size
            
            # 裁剪当前切片
            tile = img.crop((x1, y1, x2, y2))
            tile_tensor = transform(tile).unsqueeze(0).to(device)
            
            # 模型推理
            with torch.no_grad():
                tile_output = model.infer(tile_tensor)
            
            # 获取当前切片的置信度图
            tile_probs = torch.softmax(tile_output, dim=1)  # 计算每个类别的概率
            tile_confidence, _ = torch.max(tile_probs, dim=1, keepdim=True)  # 获取最高概率值
            
            # 初始化结果存储
            if final_mask is None:
                # 为整个图像分配空间
                num_classes = tile_output.shape[1]
                final_mask = torch.zeros((1, num_classes, height, width), device=device)
                confidence_map = torch.zeros((1, 1, height, width), device=device)
            
            # 更新结果，重叠区域使用置信度比较决定
            update_mask = confidence_map[:, :, y1:y2, x1:x2] < tile_confidence
            
            # 更新置信度图
            confidence_map[:, :, y1:y2, x1:x2] = torch.where(
                update_mask, 
                tile_confidence, 
                confidence_map[:, :, y1:y2, x1:x2]
            )
            
            # 更新类别预测
            # 扩展update_mask使其与类别维度匹配
            expanded_update_mask = update_mask.expand(-1, tile_output.shape[1], -1, -1)
            
            # 根据置信度更新最终掩码
            final_mask[:, :, y1:y2, x1:x2] = torch.where(
                expanded_update_mask,
                tile_output,
                final_mask[:, :, y1:y2, x1:x2]
            )
    
    return final_mask

if __name__ == '__main__':
    main()
