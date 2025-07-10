import argparse
import os

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
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
import bitsandbytes as bnb

import numpy as np
from eval_iou import SegmentationMetric

import matplotlib

from prettytable import PrettyTable
matplotlib.use('Agg')





torch.distributed.init_process_group(backend='nccl')

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def color_to_list(
    mask, palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    # mask = mask.permute(1,2,0)
    mask = mask * 255
    mask.int()
    semantic_map = np.zeros([1024, 1024], dtype=np.int8)
    for i, colour in enumerate(palette):
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map += class_map * int(i)


def onehot_to_mask(
    mask, palette=[[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 0, 0]]
):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    # x=x.permute(2,0,1)
    # x=x.numpy()
    # x = np.around
    return x


def onehot_to_index_label(mask):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    # colour_codes = np.array(palette)
    # x = np.uint8(colour_codes[x.astype(np.uint8)])*255
    # x=x.permute(2,0,1)
    # x=x.numpy()
    # x = np.around
    return x


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
        num_workers=4,
        pin_memory=False,
        sampler=sampler,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr(loader, model, config):
    # 只有rank 0 (主GPU)进行验证
    if local_rank != 0:
        # 非主GPU返回占位值
        dummy_table = None
        dummy_matrix = None
        return 0, 0, 0, 0, 'none', 'none', 'none', 'none', dummy_table, dummy_matrix
    
    model.eval()
    eval_type = config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background = config['val_dataset']['dataset']['args']['ignore_bg']
    if eval_type == 'f1':
        metric_fn = utils.calc_f1
        metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    elif eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
    elif eval_type == 'ber':
        metric_fn = utils.calc_ber
        metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    elif eval_type == 'cod':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'seg':
        metric_fn = utils.calc_cod
        metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
        
        metric_seg = SegmentationMetric(class_num, ignore_background)

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    # 显示进度条
    pbar = tqdm(total=len(loader), leave=False, desc='val')
    
    # 设备配置
    device = torch.device(f"cuda:{local_rank}")
    
    # 批量处理结果，避免频繁的CPU转换和内存占用
    batch_count = 0
    # 较大的批处理大小，因为现在只用一个GPU处理
    batch_limit = 8
    
    if eval_type == 'seg':
        # 准备标签缓存
        mask_labels = []
        gt_labels = []
    
    for batch in loader:
        batch_count += 1
        
        for k, v in batch.items():
            # 不再强制转换输入为fp16，让autocast自动管理精度
            batch[k] = v.to(device)

        inp = batch['inp']

        with torch.no_grad():
            # 使用autocast进行混合精度推理
            output_masks = model.infer(inp)
            pred = torch.sigmoid(output_masks)
        
        # 计算当前批次的指标
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])
        
        if eval_type == 'seg':
            # 处理分割数据
            for b in range(pred.shape[0]):
                output_mask = pred[b]
                gt_mask = batch['gt'][b]
                
                # 在GPU上进行处理
                mask_index = torch.argmax(output_mask, dim=0).flatten()
                gt_index = torch.argmax(gt_mask, dim=0).flatten()
                
                # 收集标签
                mask_labels.append(mask_index)
                gt_labels.append(gt_index)
            
            # 定期处理收集的标签以减少内存使用
            if len(mask_labels) >= batch_limit * inp.shape[0] or batch_count == len(loader):
                # 批量更新指标
                if mask_labels:
                    # 将标签转到CPU进行处理
                    for i in range(len(mask_labels)):
                        mask_cpu = mask_labels[i].cpu().numpy()
                        gt_cpu = gt_labels[i].cpu().numpy()
                        metric_seg.addBatch(mask_cpu, gt_cpu)
                    
                    # 清空缓存以释放GPU内存
                    mask_labels = []
                    gt_labels = []
        
        # 更新进度条
        pbar.update(1)

    pbar.close()

    # 计算最终指标
    if eval_type == 'seg':
        oa = metric_seg.overallAccuracy()
        oa = np.around(oa, decimals=4)
        mIoU, IoU = metric_seg.meanIntersectionOverUnion()
        mIoU = np.around(mIoU, decimals=4)
        IoU = np.around(IoU, decimals=4)
        
        # 处理可能出现的除零情况
        p = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=0) + 1e-10)
        p = np.around(p, decimals=4)
        mp = np.nanmean(p)
        mp = np.around(mp, decimals=4)
        
        r = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=1) + 1e-10)
        r = np.around(r, decimals=4)
        mr = np.nanmean(r)
        mr = np.around(mr, decimals=4)
        
        # 处理F1计算中的除零情况
        f1 = np.zeros_like(p)
        valid_mask = (p + r) > 0
        f1[valid_mask] = (2 * p[valid_mask] * r[valid_mask]) / (p[valid_mask] + r[valid_mask])
        f1 = np.around(f1, decimals=4)
        mf1 = np.nanmean(f1)
        mf1 = np.around(mf1, decimals=4)
        
        # 处理混淆矩阵归一化
        row_sums = metric_seg.confusionMatrix.sum(axis=0)
        valid_rows = row_sums > 0
        normed_confusionMatrix = np.zeros_like(metric_seg.confusionMatrix, dtype=float)
        normed_confusionMatrix[:, valid_rows] = metric_seg.confusionMatrix[:, valid_rows] / (row_sums[valid_rows] + 1e-10)
        normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
        
        fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
        fwIOU = np.around(fwIOU, decimals=4)
        
        classes_list = config['train_dataset']['dataset']['args']['classes']
        if ignore_background:
            axis_labels = classes_list[:-1] 
        else: 
            axis_labels = classes_list
        
        # 确保所有行的长度一致
        title_row = ['metrics', 'average']
        title_row.extend(axis_labels)
        
        # 创建表格
        table = PrettyTable(title_row)
        
        # 确保每行的长度与title_row一致
        IOU_row = ['IOU', mIoU]
        IOU_row.extend(IoU.tolist())
        # 检查并调整行长度
        if len(IOU_row) > len(title_row):
            IOU_row = IOU_row[:len(title_row)]  # 截断过长的行
        while len(IOU_row) < len(title_row):
            IOU_row.append(' ')  # 填充过短的行
            
        Precision_row = ['Precision', mp]
        Precision_row.extend(p.tolist())
        # 检查并调整行长度
        if len(Precision_row) > len(title_row):
            Precision_row = Precision_row[:len(title_row)]
        while len(Precision_row) < len(title_row):
            Precision_row.append(' ')
            
        Recall_row = ['Recall', mr]
        Recall_row.extend(r.tolist())
        # 检查并调整行长度
        if len(Recall_row) > len(title_row):
            Recall_row = Recall_row[:len(title_row)]
        while len(Recall_row) < len(title_row):
            Recall_row.append(' ')
            
        F1_row = ['F1', mf1]
        F1_row.extend(f1.tolist())
        # 检查并调整行长度
        if len(F1_row) > len(title_row):
            F1_row = F1_row[:len(title_row)]
        while len(F1_row) < len(title_row):
            F1_row.append(' ')
            
        OA_row = ['OA', oa]
        while len(OA_row) < len(title_row):
            OA_row.append(' ')
            
        fwIOU_row = ['FWIOU', fwIOU]
        while len(fwIOU_row) < len(title_row):
            fwIOU_row.append(' ')

        table.add_row(IOU_row)
        table.add_row(Precision_row)
        table.add_row(Recall_row)
        table.add_row(F1_row)
        table.add_row(OA_row)
        table.add_row(fwIOU_row)
    else:
        table = None
        normed_confusionMatrix = None

    val_metric1_avg = val_metric1.item()
    val_metric2_avg = val_metric2.item()
    val_metric3_avg = val_metric3.item()
    val_metric4_avg = val_metric4.item()
    
    # 清理GPU内存
    torch.cuda.empty_cache()
    
    return val_metric1_avg, val_metric2_avg, val_metric3_avg, val_metric4_avg, metric1, metric2, metric3, metric4, table, normed_confusionMatrix


def log_gpu_memory(stage=""):
    """记录GPU显存使用情况"""
    if local_rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        log(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def prepare_training():
    model = models.make(config['model'])
    
    # 将模型移到GPU，但保持fp32参数以支持混合精度训练
    model = model.cuda()
    log_gpu_memory("after model.cuda()")
    
    if local_rank == 0:
        log("Using mixed precision training (autocast) instead of model.half() for better gradient stability...")
    # 不再使用model.half()，而是使用autocast进行混合精度训练
    log_gpu_memory("after model setup")
    
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    log_gpu_memory("after DDP")
    
    # 使用8位优化器减少显存占用
    optimizer = bnb.optim.AdamW8bit(
        model.parameters(), 
        lr=config['optimizer']['args']['lr'], 
        betas=config['optimizer']['args'].get('betas', (0.9, 0.999)),
        eps=config['optimizer']['args'].get('eps', 1e-8),
        weight_decay=config['optimizer']['args'].get('weight_decay', 0)
    )
    log_gpu_memory("after 8-bit optimizer")
    
    # 初始化混合精度训练的GradScaler
    scaler = GradScaler()
    log_gpu_memory("after scaler")
    
    # 加载 checkpoint
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
        resume_model_path = os.path.join(
            config.get('work_dir'), 'model_epoch_' + str(config.get('resume')) + '.pth'
        )
        
        # 只在 rank 0 加载权重
        if local_rank == 0:
            checkpoint = torch.load(resume_model_path)
            if local_rank == 0:
                log(f'Loading checkpoint from {resume_model_path}')
        
        # 等待 rank 0 加载完成
        dist.barrier()
        
        # 广播权重
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
        
        # 加载权重到模型
        model.module.load_state_dict(checkpoint, strict=False)
        
        if local_rank == 0:
            log('Resume training from epoch {}'.format(epoch_start))
    else:
        epoch_start = 1
    
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model.module, optimizer, epoch_start, lr_scheduler, scaler
def train(train_loader, model, scaler):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    for batch in train_loader:
        for k, v in batch.items():
            # 不再强制转换输入为fp16，让autocast自动管理精度
            batch[k] = v.to(device)
        inp = batch['inp']
        gt = batch['gt']
        model.set_input(inp, gt)
        # 设置scaler用于混合精度训练
        model.scaler = scaler
        model.optimize_parameters()
        batch_loss_tensors = [
            torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(batch_loss_tensors, model.loss_G)
        # 立即提取标量值，避免在列表中积累张量
        loss_list.extend([loss.item() for loss in batch_loss_tensors])

        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # loss_list 现在已经包含标量，直接计算平均值
    return mean(loss_list)


# 在训练脚本中添加以下代码
def print_model_parameters(model):
    """打印模型的所有参数名称和形状"""
    table = PrettyTable(['Layer Name', 'Parameters Shape', 'Requires Grad'])
    table.align['Layer Name'] = 'l'  # 左对齐
    table.align['Parameters Shape'] = 'l'
    table.align['Requires Grad'] = 'c'  # 居中对齐
    
    for name, param in model.named_parameters():
        table.add_row([name, str(list(param.shape)), str(param.requires_grad)])
    
    if local_rank == 0:
        log('\nModel Parameters:')
        log(str(table))
        
        # 统计需要训练的参数
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log(f'\nTrainable parameters: {trainable_params:,}')
        log(f'Total parameters: {total_params:,}')
        log(f'Trainable parameters ratio: {trainable_params/total_params*100:.2f}%')

def main(config_, save_path, args):
    global config, log, writer, log_info
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

    model, optimizer, epoch_start, lr_scheduler, scaler = prepare_training()
    model.optimizer = optimizer
    lr_scheduler = CosineAnnealingLR(
        model.optimizer, config['epoch_max'], eta_min=config.get('lr_min')
    )

    # model已经在prepare_training()中处理过DDP并返回了unwrapped model


    sam_checkpoint = torch.load(config['sam_checkpoint'])
    # 自定义加载权重的函数

    def load_filtered_state_dict(model, state_dict):
        model_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape == v.shape
        }
        model.load_state_dict(filtered_state_dict, strict=False)
        unmatched_keys = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and model_dict[k].shape != v.shape
        }
        if local_rank == 0:
            log(f"warning unmatched_keys: {unmatched_keys.keys()}")


    load_filtered_state_dict(model, sam_checkpoint)
    # model.load_state_dict(sam_checkpoint, strict=False)
    
    # ===== 分层MoE权重初始化逻辑 =====
    def initialize_hierarchical_moe_from_1_5b(model, sam_checkpoint):
        """
        从1.5B模型的16个专家初始化分层MoE的6组×16个专家
        策略1：第一个专家组放置原始16个专家并冻结，其他组使用复制+噪声和专家融合
        """
        if local_rank == 0:
            log("开始初始化分层MoE权重...")
        
        # 首先检查模型是否使用分层MoE
        hierarchical_moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'use_hierarchical_moe') and module.use_hierarchical_moe:
                hierarchical_moe_layers.append(name)
        
        if len(hierarchical_moe_layers) == 0:
            if local_rank == 0:
                log("未找到分层MoE层，跳过初始化")
            return
        
        # 查找原始16个专家的权重
        original_experts = {}
        for key, value in sam_checkpoint.items():
            if 'experts.' in key and 'image_encoder' in key:
                # 解析专家索引，例如 'image_encoder.blocks.28.mlp.experts.0.0.weight'
                parts = key.split('.')
                expert_idx = None
                for i, part in enumerate(parts):
                    if part == 'experts' and i + 1 < len(parts):
                        try:
                            expert_idx = int(parts[i + 1])
                            break
                        except ValueError:
                            continue
                
                if expert_idx is not None and 0 <= expert_idx < 16:
                    # 重构key，移除layer信息，只保留专家相对路径
                    expert_key = '.'.join(parts[parts.index('experts'):])
                    original_experts[expert_key] = value
        
        if local_rank == 0:
            log(f"找到 {len(original_experts)} 个原始专家权重")
        
        if local_rank == 0:
            log(f"找到 {len(hierarchical_moe_layers)} 个分层MoE层")
        
        # 为每个分层MoE层初始化专家权重
        with torch.no_grad():
            for layer_name in hierarchical_moe_layers:
                layer_module = dict(model.named_modules())[layer_name]
                if hasattr(layer_module, 'mlp') and hasattr(layer_module.mlp, 'expert_groups'):
                    mlp_module = layer_module.mlp
                    
                    # 动态获取实际的专家组数量
                    actual_num_groups = len(mlp_module.expert_groups)
                    actual_experts_per_group = len(mlp_module.expert_groups[0]) if actual_num_groups > 0 else 0
                    
                    if local_rank == 0:
                        log(f"层 {layer_name}: 发现 {actual_num_groups} 个专家组，每组 {actual_experts_per_group} 个专家")
                    
                    # 对实际的专家组进行初始化
                    for group_idx in range(actual_num_groups):
                        expert_group = mlp_module.expert_groups[group_idx]
                        
                        if group_idx == 0:
                            # 第一个专家组：放置原始16个专家，精确复制权重（不添加噪声）
                            if local_rank == 0:
                                log(f"  初始化第一个专家组（原始专家，将被冻结）...")
                            
                            for expert_idx in range(min(actual_experts_per_group, 16)):
                                expert = expert_group[expert_idx]
                                source_expert_idx = expert_idx
                                
                                # 精确复制原始专家权重，不添加噪声
                                for orig_key, orig_weight in original_experts.items():
                                    if f'experts.{source_expert_idx}.' in orig_key:
                                        # 构建目标权重的路径
                                        target_key = orig_key.replace(f'experts.{source_expert_idx}.', '')
                                        
                                        # 获取目标模块
                                        target_parts = target_key.split('.')
                                        target_module = expert
                                        for part in target_parts[:-1]:
                                            target_module = getattr(target_module, part)
                                        
                                        param_name = target_parts[-1]
                                        if hasattr(target_module, param_name):
                                            target_param = getattr(target_module, param_name)
                                            # 精确复制，保持原始预训练权重
                                            target_param.data.copy_(orig_weight.clone())
                            
                            # 如果专家组有超过16个专家，其余的用随机初始化
                            if actual_experts_per_group > 16:
                                for expert_idx in range(16, actual_experts_per_group):
                                    expert = expert_group[expert_idx]
                                    # 使用随机初始化（小权重）
                                    for module in expert.modules():
                                        if isinstance(module, torch.nn.Linear):
                                            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
                                            if module.bias is not None:
                                                torch.nn.init.constant_(module.bias, 0.0)
                        
                        else:
                            # 其他专家组：使用复制+噪声和专家融合策略
                            if local_rank == 0:
                                log(f"  初始化第{group_idx+1}个专家组（新专家，可训练）...")
                            
                            for expert_idx in range(actual_experts_per_group):
                                expert = expert_group[expert_idx]
                                
                                if expert_idx < 16:
                                    # 复制原始专家并添加噪声
                                    source_expert_idx = expert_idx
                                    noise_std = 0.02 * group_idx  # 不同组使用不同噪声强度
                                    
                                    for orig_key, orig_weight in original_experts.items():
                                        if f'experts.{source_expert_idx}.' in orig_key:
                                            # 构建目标权重的路径
                                            target_key = orig_key.replace(f'experts.{source_expert_idx}.', '')
                                            
                                            # 获取目标模块
                                            target_parts = target_key.split('.')
                                            target_module = expert
                                            for part in target_parts[:-1]:
                                                target_module = getattr(target_module, part)
                                            
                                            param_name = target_parts[-1]
                                            if hasattr(target_module, param_name):
                                                target_param = getattr(target_module, param_name)
                                                
                                                # 复制并添加噪声
                                                copied_weight = orig_weight.clone()
                                                noise = torch.randn_like(copied_weight) * noise_std
                                                target_param.data.copy_(copied_weight + noise)
                                
                                else:
                                    # 超出原专家数量，使用专家融合
                                    expert1_idx = torch.randint(0, 16, (1,)).item()
                                    expert2_idx = torch.randint(0, 16, (1,)).item()
                                    
                                    # 随机插值权重
                                    alpha = torch.rand(1).item() * 0.6 + 0.2  # 0.2-0.8之间
                                    beta = 1.0 - alpha
                                    
                                    # 融合两个专家的权重
                                    for orig_key, orig_weight in original_experts.items():
                                        if f'experts.{expert1_idx}.' in orig_key:
                                            # 找到对应的第二个专家权重
                                            expert2_key = orig_key.replace(f'experts.{expert1_idx}.', f'experts.{expert2_idx}.')
                                            if expert2_key in original_experts:
                                                expert2_weight = original_experts[expert2_key]
                                                
                                                # 线性插值
                                                fused_weight = alpha * orig_weight + beta * expert2_weight
                                                
                                                # 设置到目标专家
                                                target_key = orig_key.replace(f'experts.{expert1_idx}.', '')
                                                target_parts = target_key.split('.')
                                                target_module = expert
                                                for part in target_parts[:-1]:
                                                    target_module = getattr(target_module, part)
                                                
                                                param_name = target_parts[-1]
                                                if hasattr(target_module, param_name):
                                                    target_param = getattr(target_module, param_name)
                                                    target_param.data.copy_(fused_weight)
                        
                        # 初始化组内门控网络
                        if group_idx < len(mlp_module.expert_gates):
                            group_gate = mlp_module.expert_gates[group_idx]
                            
                            # 使用小的随机权重初始化门控网络
                            if hasattr(group_gate, 'weight'):
                                torch.nn.init.normal_(group_gate.weight, mean=0.0, std=0.01)
                            if hasattr(group_gate, 'bias') and group_gate.bias is not None:
                                torch.nn.init.constant_(group_gate.bias, 0.0)
                    
                    # 初始化第一级门控网络（专家组选择）
                    if hasattr(mlp_module, 'group_gate'):
                        torch.nn.init.normal_(mlp_module.group_gate.weight, mean=0.0, std=0.01)
                        if mlp_module.group_gate.bias is not None:
                            torch.nn.init.constant_(mlp_module.group_gate.bias, 0.0)
        
        # 冻结第一个专家组的权重（原始专家）
        if local_rank == 0:
            log("冻结第一个专家组的权重（原始1.5B专家）...")
        
        frozen_params = 0
        for layer_name in hierarchical_moe_layers:
            layer_module = dict(model.named_modules())[layer_name]
            if hasattr(layer_module, 'mlp') and hasattr(layer_module.mlp, 'expert_groups'):
                mlp_module = layer_module.mlp
                
                # 冻结第一个专家组的所有参数
                if len(mlp_module.expert_groups) > 0:
                    first_expert_group = mlp_module.expert_groups[0]
                    for expert in first_expert_group:
                        for param in expert.parameters():
                            param.requires_grad = False
                            frozen_params += param.numel()
        
        if local_rank == 0:
            log(f"已冻结 {frozen_params:,} 个原始专家参数")
            log("分层MoE权重初始化完成")
    
    # 调用权重初始化函数
    if config['model']['name'] == 'sam_hierarchical_moe_10b' or 'hierarchical_moe' in config['model']['name']:
        initialize_hierarchical_moe_from_1_5b(model, sam_checkpoint)

    print_model_parameters(model)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        # print(
        #     'model_grad_params:' + str(model_grad_params),
        #     '\nmodel_total_params:' + str(model_total_params),
        # )
        # log('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))

        # 打印所有需要梯度更新的层的名字
        for name, para in model.named_parameters():
            if para.requires_grad:
                log(f'u are train {name}')
        log(
            'model_grad_params:'
            + str(model_grad_params)
            + '\nmodel_total_params:'
            + str(model_total_params)
        )

    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18 if config['eval_type'] != 'ber' else 1e8
    timer = utils.Timer()
    # 在首次训练前记录显存使用
    if local_rank == 0:
        log_gpu_memory("before training start")
    
    for epoch in range(epoch_start, epoch_max + 1):
        if train_loader is not None and hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, scaler)
        lr_scheduler.step()
        
        # 每个epoch后清理显存并记录
        torch.cuda.empty_cache()
        if epoch == epoch_start and local_rank == 0:
            log_gpu_memory(f"after first epoch {epoch}")

        if local_rank == 0:
            log_info = [
                '\n ############################ epoch {}/{} ############################'.format(
                    epoch, epoch_max
                )
            ]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            log_info.append('train G: loss={:.4f}'.format(train_loss_G))
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            model_spec = config['model']
            model_spec['sd'] = model.state_dict()
            optimizer_spec = config['optimizer']
            optimizer_spec['sd'] = optimizer.state_dict()

            save(config, model, save_path, 'last')

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            with torch.no_grad():
                (
                    result1,
                    result2,
                    result3,
                    result4,
                    metric1,
                    metric2,
                    metric3,
                    metric4,
                    seg_eval_table,
                    normed_confusionMatrix,
                ) = eval_psnr(val_loader, model, config)
                # eval_type=config.get('eval_type'))

            if local_rank == 0:
                save(config, model, save_path, str(epoch))

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best')
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best')

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append(
                    'epoch train + val time: {} {}/{}'.format(t_epoch, t_elapsed, t_all)
                )

                log_info.append(str(seg_eval_table))
                log_info.append('Confusion Matrix:')
                log_info.append(str(normed_confusionMatrix))

                log('\n'.join(log_info))
                writer.flush()


def save(config, model, save_path, name):
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            torch.save(
                {"prompt": prompt_generator, "decode_head": decode_head},
                os.path.join(save_path, f"prompt_epoch_{name}.pth"),
            )
        else:
            torch.save(
                model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")
            )
    else:
        torch.save(
            model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default="configs/train/setr/train_setr_evp_cod.yaml"
    )
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
        save_name = args.config.split('/')[-1][: -len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
