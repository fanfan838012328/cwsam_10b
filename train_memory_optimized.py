"""
内存优化的CWSAM训练脚本

该脚本针对内存受限的环境进行了优化，包括：
- 梯度检查点
- 混合精度训练
- 内存清理
- 批次累积
"""

import argparse
import os
import gc
import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from eval_iou import SegmentationMetric

import matplotlib
from prettytable import PrettyTable
matplotlib.use('Agg')

# 设置内存优化选项
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 初始化分布式训练
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend='nccl')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# 设置CUDA内存分配策略
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def clear_memory():
    """清理GPU内存"""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def make_data_loader(spec, tag=''):
    if spec is None:
        return None

    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    if local_rank == 0:
        log('{} dataset: size={}'.format(tag, len(dataset)))
        for k, v in dataset[0].items():
            log('  {}: shape={}'.format(k, tuple(v.shape)))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=2,  # 减少worker数量
        pin_memory=True,
        sampler=sampler,
        drop_last=True,  # 丢弃最后一个不完整的batch
    )
    return loader

def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr(loader, model, config):
    if local_rank != 0:
        dummy_table = None
        dummy_matrix = None
        return 0, 0, 0, 0, 'none', 'none', 'none', 'none', dummy_table, dummy_matrix
    
    model.eval()
    eval_type = config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background = config['val_dataset']['dataset']['args']['ignore_bg']
    
    if eval_type == 'seg':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
        metric_seg = SegmentationMetric(class_num, ignore_background)

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(total=len(loader), leave=False, desc='val')
    device = torch.device(f"cuda:{local_rank}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            for k, v in batch.items():
                batch[k] = v.to(device, non_blocking=True)

            inp = batch['inp']
            
            # 使用混合精度推理
            with autocast():
                output_masks = model.infer(inp)
                pred = torch.sigmoid(output_masks)
            
            # 计算指标
            result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
            val_metric1.add(result1.item(), inp.shape[0])
            val_metric2.add(result2.item(), inp.shape[0])
            val_metric3.add(result3.item(), inp.shape[0])
            val_metric4.add(result4.item(), inp.shape[0])
            
            if eval_type == 'seg':
                for b in range(pred.shape[0]):
                    output_mask = pred[b]
                    gt_mask = batch['gt'][b]
                    
                    mask_index = torch.argmax(output_mask, dim=0).cpu().numpy()
                    gt_index = torch.argmax(gt_mask, dim=0).cpu().numpy()
                    
                    metric_seg.addBatch(mask_index.flatten(), gt_index.flatten())
            
            # 定期清理内存
            if batch_idx % 10 == 0:
                clear_memory()
            
            pbar.update(1)

    pbar.close()
    clear_memory()

    # 计算最终指标
    if eval_type == 'seg':
        oa = metric_seg.overallAccuracy()
        mIoU, IoU = metric_seg.meanIntersectionOverUnion()
        
        # 创建简化的表格
        table = PrettyTable(['Metric', 'Value'])
        table.add_row(['OA', f'{oa:.4f}'])
        table.add_row(['mIoU', f'{mIoU:.4f}'])
        
        normed_confusionMatrix = metric_seg.confusionMatrix
    else:
        table = None
        normed_confusionMatrix = None

    return (val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item(),
            metric1, metric2, metric3, metric4, table, normed_confusionMatrix)

def prepare_training():
    model = models.make(config['model'])
    
    # 启用梯度检查点
    if hasattr(model, 'image_encoder'):
        model.image_encoder.gradient_checkpointing = True
    
    model = model.cuda()
    
    # 使用更保守的DDP设置
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # 设为False以提高性能
        broadcast_buffers=False,
        bucket_cap_mb=25,  # 减小bucket大小
    )
    
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    
    # 加载checkpoint
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
        resume_model_path = os.path.join(
            config.get('work_dir'), 'model_epoch_' + str(config.get('resume')) + '.pth'
        )
        
        if local_rank == 0:
            checkpoint = torch.load(resume_model_path, map_location='cpu')
            log(f'Loading checkpoint from {resume_model_path}')
        
        dist.barrier()
        
        if local_rank == 0:
            for k, v in checkpoint.items():
                v = v.cuda()
                dist.broadcast(v, 0)
                checkpoint[k] = v
        else:
            checkpoint = {}
            for k, _ in model.module.state_dict().items():
                v = torch.empty_like(model.module.state_dict()[k]).cuda()
                dist.broadcast(v, 0)
                checkpoint[k] = v
        
        model.module.load_state_dict(checkpoint, strict=False)
        
        if local_rank == 0:
            log('Resume training from epoch {}'.format(epoch_start))
    else:
        epoch_start = 1
    
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model.module, optimizer, epoch_start, lr_scheduler

def train_with_accumulation(train_loader, model, accumulation_steps=4):
    """使用梯度累积的训练函数"""
    model.train()
    
    # 初始化混合精度训练
    scaler = GradScaler()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    model.optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)
        
        inp = batch['inp']
        gt = batch['gt']
        
        # 使用混合精度训练
        with autocast():
            model.set_input(inp, gt)
            model.forward()
            loss = model.loss_G / accumulation_steps  # 缩放损失
        
        # 反向传播
        scaler.scale(loss).backward()
        
        # 梯度累积
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(model.optimizer)
            scaler.update()
            model.optimizer.zero_grad()
            
            # 收集损失
            batch_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            dist.all_gather(batch_loss, loss)
            loss_list.extend([l.item() * accumulation_steps for l in batch_loss])
        
        # 定期清理内存
        if batch_idx % 20 == 0:
            clear_memory()
        
        if pbar is not None:
            pbar.update(1)

    # 处理剩余的梯度
    if len(train_loader) % accumulation_steps != 0:
        scaler.step(model.optimizer)
        scaler.update()
        model.optimizer.zero_grad()

    if pbar is not None:
        pbar.close()
    
    clear_memory()
    return mean(loss_list) if loss_list else 0.0

def main(config_, save_path, args):
    global config, log, writer
    config = config_
    log, writer = utils.set_save_path(save_path, remove=False)
    
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]},
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()
    model.optimizer = optimizer
    
    # 加载SAM预训练权重
    if config.get('sam_checkpoint'):
        sam_checkpoint = torch.load(config['sam_checkpoint'], map_location='cpu')
        
        def load_filtered_state_dict(model, state_dict):
            model_dict = model.state_dict()
            filtered_state_dict = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            model.load_state_dict(filtered_state_dict, strict=False)
            
            if local_rank == 0:
                unmatched_keys = {
                    k: v for k, v in state_dict.items()
                    if k in model_dict and model_dict[k].shape != v.shape
                }
                log(f"Warning unmatched_keys: {list(unmatched_keys.keys())}")

        load_filtered_state_dict(model, sam_checkpoint)
        del sam_checkpoint
        clear_memory()

    # 设置参数训练策略
    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "image_encoder" in name and "Adapter" in name:
            para.requires_grad_(True)
        if "image_encoder" in name and "experts" in name:
            para.requires_grad_(True)
        if "image_encoder" in name and "gate" in name:
            para.requires_grad_(True)

    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f'Total params: {model_total_params:,}')
        log(f'Trainable params: {model_grad_params:,}')
        log(f'Trainable ratio: {model_grad_params/model_total_params*100:.2f}%')

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val', 5)
    epoch_save = config.get('epoch_save', 10)
    max_val_v = -1e18
    timer = utils.Timer()
    
    # 梯度累积步数
    accumulation_steps = config.get('accumulation_steps', 4)
    
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        
        # 使用梯度累积训练
        train_loss_G = train_with_accumulation(train_loader, model, accumulation_steps)
        lr_scheduler.step()

        if local_rank == 0:
            log_info = [f'\n ############################ epoch {epoch}/{epoch_max} ############################']
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            # 保存最新模型
            torch.save(model.state_dict(), os.path.join(save_path, 'model_epoch_last.pth'))

        # 验证
        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                (result1, result2, result3, result4, metric1, metric2, metric3, metric4,
                 seg_eval_table, normed_confusionMatrix) = eval_psnr(val_loader, model, config)

            if local_rank == 0:
                if epoch % epoch_save == 0:
                    torch.save(model.state_dict(), os.path.join(save_path, f'model_epoch_{epoch}.pth'))

                if result1 > max_val_v:
                    max_val_v = result1
                    torch.save(model.state_dict(), os.path.join(save_path, 'model_epoch_best.pth'))

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append(f'epoch time: {t_epoch} {t_elapsed}/{t_all}')

                if seg_eval_table:
                    log_info.append(str(seg_eval_table))

                log('\n'.join(log_info))
                writer.flush()
        
        # 定期清理内存
        clear_memory()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="configs/train/sam/train_sam_moe_1dot5b_optimized.yaml")
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)