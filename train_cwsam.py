"""
CWSAM模型训练脚本

该脚本用于训练不同规模的CWSAM模型，支持1.5B、2B、3B、5B、7B和10B规模。
"""

import argparse
import os
import yaml
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description='CWSAM模型训练脚本')
    parser.add_argument('--model_size', type=str, default='3B', 
                        choices=['1.5B', '2B', '3B', '5B', '7B', '10B'],
                        help='模型规模 (1.5B, 2B, 3B, 5B, 7B, 10B)')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='训练批次大小，如果不指定则使用配置文件中的默认值')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数，如果不指定则使用配置文件中的默认值')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率，如果不指定则使用配置文件中的默认值')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='预训练权重路径，如果不指定则使用配置文件中的默认值')
    parser.add_argument('--tag', type=str, default=None,
                        help='实验标签，用于区分不同的训练运行')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='分布式训练的本地排名')
    args = parser.parse_args()
    
    # 根据模型规模选择配置文件
    model_size_map = {
        '1.5B': 'configs/train/sam/train_sam_moe_1dot5b.yaml',
        '2B': 'configs/train/sam/train_sam_moe_2b.yaml',
        '3B': 'configs/train/sam/train_sam_moe_3b.yaml',
        '5B': 'configs/train/sam/train_sam_moe_5b.yaml',
        '7B': 'configs/train/sam/train_sam_moe_7b.yaml',
        '10B': 'configs/train/sam/train_sam_moe_10b.yaml'
    }
    
    config_path = model_size_map.get(args.model_size)
    if not config_path or not os.path.exists(config_path):
        print(f"错误: 找不到模型规模 {args.model_size} 的配置文件 {config_path}")
        return
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 更新配置（如果指定了命令行参数）
    if args.batch_size is not None:
        config['train_dataset']['batch_size'] = args.batch_size
        config['val_dataset']['batch_size'] = max(1, args.batch_size // 2)
    
    if args.epochs is not None:
        config['epoch_max'] = args.epochs
    
    if args.lr is not None:
        config['optimizer']['args']['lr'] = args.lr
    
    if args.checkpoint is not None:
        config['sam_checkpoint'] = args.checkpoint
    
    # 创建临时配置文件
    temp_config_path = f"configs/train/sam/temp_{args.model_size.lower()}.yaml"
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    
    # 构建训练命令
    save_name = f"cwsam_{args.model_size.lower()}"
    if args.tag:
        save_name += f"_{args.tag}"
    
    # 使用train1.py进行训练
    cmd_args = [
        f"--config {temp_config_path}",
        f"--name {save_name}",
        f"--local_rank {args.local_rank}"
    ]
    
    # 导入并运行train1.py中的main函数
    import sys
    sys.path.append('.')
    from train1 import main
    
    # 创建保存路径
    save_path = os.path.join('./save', save_name)
    
    # 运行训练
    main(config, save_path, args)
    
    # 清理临时配置文件
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)

if __name__ == '__main__':
    # 初始化分布式训练
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend='nccl')
    
    main()