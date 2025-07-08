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

def prepare_image(image_path, input_size=1024):
    """准备输入图像"""
    # 使用 PIL.Image.BILINEAR 进行更快的缩放
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 读取并处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加batch维度

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                       default='/public/home/daiwenxuan/project/fan/cwsam/configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml', 
                       help='配置文件路径')
    parser.add_argument('--model', 
                       default='/public/home/daiwenxuan/project/fan/cwsam/save/XinTong_sam_vit_h_moe_3b/model_epoch_121.pth',
                       help='模型权重路径')
    parser.add_argument('--input_dir', 
                       default='/public/home/daiwenxuan/project/fan/data/dataset/XinTong/train_100',
                       help='输入图像文件夹')
    parser.add_argument('--output_dir', 
                       default='/public/home/daiwenxuan/project/fan/cwsam/output/train_100img_121',
                       help='输出结果文件夹')
    parser.add_argument('--device', 
                       default='cuda',
                       choices=['cuda', 'cpu'],
                       help='选择推理设备 (cuda/cpu)')
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
    BATCH_SIZE = 2 if device.type == 'cuda' else 2  # CPU时使用较小的批量
    
    # 处理每张图像
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="处理图像批次"):
        batch_files = image_files[i:i + BATCH_SIZE]
        batch_inputs = []
        
        # 准备批处理输入
        for image_file in batch_files:
            image_path = os.path.join(args.input_dir, image_file)
            input_tensor = prepare_image(image_path)
            batch_inputs.append(input_tensor)
            
        batch_inputs = torch.cat(batch_inputs, dim=0).to(device)
        
        # 批量推理
        with torch.no_grad():
            batch_outputs = model.infer(batch_inputs)
            # 获取预测概率
            batch_probs = torch.softmax(batch_outputs, dim=1)
            # 获取最大概率值
            confidence, _ = torch.max(batch_probs, dim=1)
            
        # 处理每个输出
        for idx, image_file in enumerate(batch_files):
            output_masks = batch_outputs[idx].cpu()
            # confidence_map = confidence[idx].cpu().numpy()
            
            # 转换为彩色掩码
            binary_mask = onehot_to_mask(output_masks, palette=color_palette)
            
            # 保存掩码
            mask_path = os.path.join(args.output_dir, 'masks', f'{os.path.splitext(image_file)[0]}_mask.png')
            Image.fromarray(np.uint8(binary_mask)).convert('RGB').save(mask_path)
            
            # # 保存置信度图
            # confidence_path = os.path.join(args.output_dir, 'masks', f'{os.path.splitext(image_file)[0]}_confidence.npy')
            # np.save(confidence_path, confidence_map)

if __name__ == '__main__':
    main()
