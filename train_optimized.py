import argparse
import os
import random

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
from torch.cuda.amp import autocast, GradScaler


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
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x


def onehot_to_index_label(mask):
    mask = mask.permute(1, 2, 0).numpy()
    x = np.argmax(mask, axis=-1)
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
    
    # 优化: 减少num_workers和启用pin_memory
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=8,  # 减少worker数量以降低内存占用
        pin_memory=True,  # 启用pin_memory
        sampler=sampler,
        persistent_workers=False,  # 禁用持久worker以减少内存
        prefetch_factor=2,  # 减少预取批次数
    )
    return loader


def multi_task_collate_fn(batch):
    """优化的collate函数，减少内存分配"""
    task_groups = {}
    for item in batch:
        task_id = item['task_id'].item() if hasattr(item['task_id'], 'item') else item['task_id']
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(item)
    
    max_classes = max(items[0]['gt'].shape[0] for items in task_groups.values())
    
    batched_data = {'inp': [], 'gt': [], 'task_id': []}
    
    for task_id, task_items in task_groups.items():
        task_inp = torch.stack([item['inp'] for item in task_items])
        task_gt = torch.stack([item['gt'] for item in task_items])
        task_ids = torch.full((len(task_items),), task_id, dtype=torch.long)
        
        current_classes = task_gt.shape[1]
        if current_classes < max_classes:
            padding_size = max_classes - current_classes
            padding = torch.zeros(task_gt.shape[0], padding_size, task_gt.shape[2], task_gt.shape[3], 
                                 dtype=task_gt.dtype, device=task_gt.device)
            task_gt = torch.cat([task_gt, padding], dim=1)
            del padding
        
        batched_data['inp'].append(task_inp)
        batched_data['gt'].append(task_gt)
        batched_data['task_id'].append(task_ids)
        
        # 即时清理不需要的临时变量
        del task_inp, task_gt, task_ids
    
    final_inp = torch.cat(batched_data['inp'], dim=0)
    final_gt = torch.cat(batched_data['gt'], dim=0)
    final_task_ids = torch.cat(batched_data['task_id'], dim=0)
    
    # 清理中间变量
    del batched_data
    
    return {
        'inp': final_inp,
        'gt': final_gt,
        'task_id': final_task_ids
    }


def make_multi_data_loader(datasets_config, tag=''):
    if datasets_config is None:
        return None
    
    all_datasets = []
    total_size = 0
    batch_size = 1
    
    for dataset_config in datasets_config:
        dataset = datasets.make(dataset_config['dataset'])
        dataset = datasets.make(dataset_config['wrapper'], args={'dataset': dataset})
        all_datasets.append(dataset)
        total_size += len(dataset)
        
        if 'batch_size' in dataset_config:
            batch_size = dataset_config['batch_size']
        
        if local_rank == 0:
            log('{} dataset (task_id={}): size={}'.format(
                tag, dataset_config['dataset']['args'].get('task_id', 'unknown'), len(dataset)))
            
            sample = dataset[0]
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    log('  {}: shape={}'.format(k, tuple(v.shape)))
                else:
                    log('  {}: {}'.format(k, v))
    
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    if local_rank == 0:
        log('{} combined dataset: total_size={}, batch_size={}'.format(tag, total_size, batch_size))
    
    sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset)
    
    # 优化的DataLoader
    loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=4,  # 减少worker数量以降低内存占用
        pin_memory=True,  # 启用pin_memory
        sampler=sampler,
        collate_fn=multi_task_collate_fn,
        persistent_workers=False,  # 禁用持久worker以减少内存
        prefetch_factor=1,  # 减少预取批次数
    )
    return loader


def make_data_loaders():
    if 'train_datasets' in config:
        train_loader = make_multi_data_loader(config.get('train_datasets'), tag='train')
    else:
        train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    
    if 'val_datasets' in config:
        val_loader = make_multi_data_loader(config.get('val_datasets'), tag='val')
    else:
        val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    
    return train_loader, val_loader


def eval_psnr(loader, model, config):
    if local_rank != 0:
        dummy_table = None
        dummy_matrix = None
        return 0, 0, 0, 0, 'none', 'none', 'none', 'none', dummy_table, dummy_matrix
    
    model.eval()
    eval_type = config.get('eval_type')
    
    # 获取任务配置信息
    task_configs = None
    if 'task_configs' in config['model']['args']:
        task_configs = config['model']['args']['task_configs']
        class_num = max(task_config['num_classes'] for task_config in task_configs)
    else:
        class_num = config['model']['args']['num_classes']
    
    if 'val_datasets' in config:
        ignore_background = config['val_datasets'][0]['dataset']['args']['ignore_bg']
    else:
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
        # 为多任务创建独立的SegmentationMetric
        if task_configs:
            metric_segs = {}
            for task_config in task_configs:
                task_id = task_config['task_id']
                task_class_num = task_config['num_classes']
                metric_segs[task_id] = SegmentationMetric(task_class_num, ignore_background)
        else:
            metric_seg = SegmentationMetric(class_num, ignore_background)

    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    pbar = tqdm(total=len(loader), leave=False, desc='val')
    device = torch.device(f"cuda:{local_rank}")
    
    batch_count = 0
    batch_limit = 8
    
    if eval_type == 'seg':
        # 为多任务创建独立的mask和gt标签收集器
        if task_configs:
            task_mask_labels = {}
            task_gt_labels = {}
            for task_config in task_configs:
                task_id = task_config['task_id']
                task_mask_labels[task_id] = []
                task_gt_labels[task_id] = []
        else:
            mask_labels = []
            gt_labels = []
    
    for batch in loader:
        batch_count += 1
        
        for k, v in batch.items():
            batch[k] = v.to(device)

        inp = batch['inp']

        with torch.no_grad():
            with autocast():  # 使用混合精度进行推理
                if 'task_id' in batch:
                    task_id = batch['task_id'][0].item()
                    output_masks = model.infer(inp, task_id=task_id)
                else:
                    output_masks = model.infer(inp)
                pred = torch.sigmoid(output_masks)
        
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])
        
        if eval_type == 'seg':
            for b in range(pred.shape[0]):
                output_mask = pred[b]
                gt_mask = batch['gt'][b]
                
                # 获取当前样本的任务ID
                if 'task_id' in batch and task_configs:
                    current_task_id = batch['task_id'][b].item()
                    # 获取对应任务的真实类别数
                    actual_classes = None
                    for task_config in task_configs:
                        if task_config['task_id'] == current_task_id:
                            actual_classes = task_config['num_classes']
                            break
                    
                    if actual_classes is not None:
                        # 只处理该任务的实际类别数
                        output_mask = output_mask[:actual_classes]
                        gt_mask = gt_mask[:actual_classes]
                        
                        mask_index = torch.argmax(output_mask, dim=0).flatten()
                        gt_index = torch.argmax(gt_mask, dim=0).flatten()
                        
                        task_mask_labels[current_task_id].append(mask_index)
                        task_gt_labels[current_task_id].append(gt_index)
                else:
                    # 单任务处理
                    mask_index = torch.argmax(output_mask, dim=0).flatten()
                    gt_index = torch.argmax(gt_mask, dim=0).flatten()
                    
                    mask_labels.append(mask_index)
                    gt_labels.append(gt_index)
            
            # 批量处理累积的标签 - 多任务版本
            if task_configs:
                for task_id in task_mask_labels:
                    if len(task_mask_labels[task_id]) >= batch_limit or batch_count == len(loader):
                        if task_mask_labels[task_id]:
                            for i in range(len(task_mask_labels[task_id])):
                                mask_cpu = task_mask_labels[task_id][i].cpu().numpy()
                                gt_cpu = task_gt_labels[task_id][i].cpu().numpy()
                                metric_segs[task_id].addBatch(mask_cpu, gt_cpu)
                            
                            task_mask_labels[task_id] = []
                            task_gt_labels[task_id] = []
            else:
                # 单任务版本
                if len(mask_labels) >= batch_limit * inp.shape[0] or batch_count == len(loader):
                    if mask_labels:
                        for i in range(len(mask_labels)):
                            mask_cpu = mask_labels[i].cpu().numpy()
                            gt_cpu = gt_labels[i].cpu().numpy()
                            metric_seg.addBatch(mask_cpu, gt_cpu)
                        
                        mask_labels = []
                        gt_labels = []
        
        pbar.update(1)
        
        # 内存清理 - 删除不需要的变量
        del inp, output_masks, pred
        del result1, result2, result3, result4
        del batch
        
        # 定期清理CUDA缓存
        if batch_count % 20 == 0:  # 每20个batch清理一次
            torch.cuda.empty_cache()

    pbar.close()
    
    # 验证结束后清理内存
    torch.cuda.empty_cache()

    if eval_type == 'seg':
        if task_configs:
            # 多任务：为每个任务生成独立的评估表格
            tables = []
            confusion_matrices = []
            
            # 获取每个任务的类别列表
            task_classes = {}
            if 'val_datasets' in config:
                for i, dataset_config in enumerate(config['val_datasets']):
                    if 'task_id' in dataset_config['dataset']['args']:
                        task_id = dataset_config['dataset']['args']['task_id']
                        task_classes[task_id] = dataset_config['dataset']['args']['classes']
                    else:
                        # 如果没有task_id，默认为0
                        task_classes[0] = dataset_config['dataset']['args']['classes']
            
            for task_config in task_configs:
                task_id = task_config['task_id']
                task_name = task_config.get('name', f'Task_{task_id}')
                
                if task_id not in metric_segs:
                    continue
                    
                metric_seg = metric_segs[task_id]
                
                oa = metric_seg.overallAccuracy()
                oa = np.around(oa, decimals=4)
                mIoU, IoU = metric_seg.meanIntersectionOverUnion()
                mIoU = np.around(mIoU, decimals=4)
                IoU = np.around(IoU, decimals=4)
                
                p = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=0) + 1e-10)
                p = np.around(p, decimals=4)
                mp = np.nanmean(p)
                mp = np.around(mp, decimals=4)
                
                r = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=1) + 1e-10)
                r = np.around(r, decimals=4)
                mr = np.nanmean(r)
                mr = np.around(mr, decimals=4)
                
                f1 = np.zeros_like(p)
                valid_mask = (p + r) > 0
                f1[valid_mask] = (2 * p[valid_mask] * r[valid_mask]) / (p[valid_mask] + r[valid_mask])
                f1 = np.around(f1, decimals=4)
                mf1 = np.nanmean(f1)
                mf1 = np.around(mf1, decimals=4)
                
                row_sums = metric_seg.confusionMatrix.sum(axis=0)
                valid_rows = row_sums > 0
                normed_confusionMatrix = np.zeros_like(metric_seg.confusionMatrix, dtype=float)
                normed_confusionMatrix[:, valid_rows] = metric_seg.confusionMatrix[:, valid_rows] / (row_sums[valid_rows] + 1e-10)
                normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
                
                fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
                fwIOU = np.around(fwIOU, decimals=4)
                
                # 获取该任务的类别列表
                classes_list = task_classes.get(task_id, [f'class_{i}' for i in range(task_config['num_classes'])])
                
                if ignore_background:
                    axis_labels = classes_list[:-1] 
                else: 
                    axis_labels = classes_list
                
                title_row = ['metrics', 'average']
                title_row.extend(axis_labels)
                
                table = PrettyTable(title_row)
                table.title = f"{task_name} Evaluation Results"
                
                IOU_row = ['IOU', mIoU]
                IOU_row.extend(IoU.tolist())
                if len(IOU_row) > len(title_row):
                    IOU_row = IOU_row[:len(title_row)]
                while len(IOU_row) < len(title_row):
                    IOU_row.append(' ')
                    
                Precision_row = ['Precision', mp]
                Precision_row.extend(p.tolist())
                if len(Precision_row) > len(title_row):
                    Precision_row = Precision_row[:len(title_row)]
                while len(Precision_row) < len(title_row):
                    Precision_row.append(' ')
                    
                Recall_row = ['Recall', mr]
                Recall_row.extend(r.tolist())
                if len(Recall_row) > len(title_row):
                    Recall_row = Recall_row[:len(title_row)]
                while len(Recall_row) < len(title_row):
                    Recall_row.append(' ')
                    
                F1_row = ['F1', mf1]
                F1_row.extend(f1.tolist())
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
                
                tables.append(table)
                confusion_matrices.append(normed_confusionMatrix)
            
            # 返回第一个任务的结果作为主要指标，但包含所有表格
            if tables:
                table = tables
                normed_confusionMatrix = confusion_matrices
            else:
                table = None
                normed_confusionMatrix = None
        else:
            # 单任务处理
            oa = metric_seg.overallAccuracy()
            oa = np.around(oa, decimals=4)
            mIoU, IoU = metric_seg.meanIntersectionOverUnion()
            mIoU = np.around(mIoU, decimals=4)
            IoU = np.around(IoU, decimals=4)
            
            p = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=0) + 1e-10)
            p = np.around(p, decimals=4)
            mp = np.nanmean(p)
            mp = np.around(mp, decimals=4)
            
            r = np.diag(metric_seg.confusionMatrix) / (metric_seg.confusionMatrix.sum(axis=1) + 1e-10)
            r = np.around(r, decimals=4)
            mr = np.nanmean(r)
            mr = np.around(mr, decimals=4)
            
            f1 = np.zeros_like(p)
            valid_mask = (p + r) > 0
            f1[valid_mask] = (2 * p[valid_mask] * r[valid_mask]) / (p[valid_mask] + r[valid_mask])
            f1 = np.around(f1, decimals=4)
            mf1 = np.nanmean(f1)
            mf1 = np.around(mf1, decimals=4)
            
            row_sums = metric_seg.confusionMatrix.sum(axis=0)
            valid_rows = row_sums > 0
            normed_confusionMatrix = np.zeros_like(metric_seg.confusionMatrix, dtype=float)
            normed_confusionMatrix[:, valid_rows] = metric_seg.confusionMatrix[:, valid_rows] / (row_sums[valid_rows] + 1e-10)
            normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
            
            fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
            fwIOU = np.around(fwIOU, decimals=4)
            
            if 'val_datasets' in config:
                classes_list = config['val_datasets'][0]['dataset']['args']['classes']
            else:
                classes_list = config['val_dataset']['dataset']['args']['classes']
                
            if ignore_background:
                axis_labels = classes_list[:-1] 
            else: 
                axis_labels = classes_list
            
            title_row = ['metrics', 'average']
            title_row.extend(axis_labels)
            
            table = PrettyTable(title_row)
            
            IOU_row = ['IOU', mIoU]
            IOU_row.extend(IoU.tolist())
            if len(IOU_row) > len(title_row):
                IOU_row = IOU_row[:len(title_row)]
            while len(IOU_row) < len(title_row):
                IOU_row.append(' ')
                
            Precision_row = ['Precision', mp]
            Precision_row.extend(p.tolist())
            if len(Precision_row) > len(title_row):
                Precision_row = Precision_row[:len(title_row)]
            while len(Precision_row) < len(title_row):
                Precision_row.append(' ')
                
            Recall_row = ['Recall', mr]
            Recall_row.extend(r.tolist())
            if len(Recall_row) > len(title_row):
                Recall_row = Recall_row[:len(title_row)]
            while len(Recall_row) < len(title_row):
                Recall_row.append(' ')
                
            F1_row = ['F1', mf1]
            F1_row.extend(f1.tolist())
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
    
    torch.cuda.empty_cache()
    
    return val_metric1_avg, val_metric2_avg, val_metric3_avg, val_metric4_avg, metric1, metric2, metric3, metric4, table, normed_confusionMatrix


def prepare_training():
    model = models.make(config['model'])
    
    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    
    optimizer = utils.make_optimizer(model.parameters(), config['optimizer'])
    
    max_epoch = config.get('epoch_max')
    lr_scheduler = CosineAnnealingLR(optimizer, max_epoch, eta_min=config.get('lr_min'))
    
    # 创建GradScaler用于混合精度训练
    scaler = GradScaler()
    
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
        resume_model_path = os.path.join(
            config.get('work_dir'), 'model_epoch_' + str(config.get('resume')) + '.pth'
        )
        
        # 初始化checkpoint变量
        checkpoint = None
        
        if local_rank == 0:
            checkpoint = torch.load(resume_model_path, map_location='cpu')
            log(f'Loading checkpoint from {resume_model_path}')
            
            # 检查checkpoint格式
            if 'model_state_dict' in checkpoint:
                # 新格式：包含完整训练状态
                log('Loading full training state...')
                
                # 恢复随机数状态
                if 'random_states' in checkpoint:
                    random.setstate(checkpoint['random_states']['python'])
                    np.random.set_state(checkpoint['random_states']['numpy'])
                    torch.set_rng_state(checkpoint['random_states']['torch'])
                    if checkpoint['random_states']['torch_cuda'] and torch.cuda.is_available():
                        torch.cuda.set_rng_state_all(checkpoint['random_states']['torch_cuda'])
                
                model_state = checkpoint['model_state_dict']
            else:
                # 旧格式：只有模型权重
                log('Loading legacy checkpoint (model weights only)...')
                model_state = checkpoint
        
        dist.barrier()
        
        if local_rank == 0:
            # 广播模型状态
            for k, v in model_state.items():
                if not v.is_cuda:
                    v = v.cuda()
                dist.broadcast(v, 0)
                model_state[k] = v
        else:
            # 接收模型状态
            model_state = {}
            model_dict = model.module.state_dict()
            for k in model_dict.keys():
                v = torch.empty_like(model_dict[k]).cuda()
                dist.broadcast(v, 0)
                model_state[k] = v
        
        # 加载模型权重
        model.module.load_state_dict(model_state, strict=False)
        
        # 恢复优化器状态（只在rank 0处理）
        if local_rank == 0 and checkpoint is not None and 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                log('Optimizer state restored')
            except Exception as e:
                log(f'Warning: Could not restore optimizer state: {e}')
        
        # 恢复学习率调度器状态
        if local_rank == 0 and checkpoint is not None and 'model_state_dict' in checkpoint and 'lr_scheduler_state_dict' in checkpoint:
            try:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
                log('Learning rate scheduler state restored')
            except Exception as e:
                log(f'Warning: Could not restore lr_scheduler state: {e}')
        
        # 恢复混合精度缩放器状态
        if local_rank == 0 and checkpoint is not None and 'model_state_dict' in checkpoint and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                log('Mixed precision scaler state restored')
            except Exception as e:
                log(f'Warning: Could not restore scaler state: {e}')
        
        # 恢复epoch信息
        if local_rank == 0 and checkpoint is not None and 'model_state_dict' in checkpoint and 'epoch' in checkpoint:
            epoch_start = checkpoint['epoch'] + 1
            log(f'Resuming from epoch {checkpoint["epoch"]}, will start at epoch {epoch_start}')
        
        if local_rank == 0:
            log('Resume training from epoch {}'.format(epoch_start))
    else:
        epoch_start = 1
    
    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model.module, optimizer, epoch_start, lr_scheduler, scaler


def train(train_loader, model, scaler):
    """优化的训练函数"""
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    
    for idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device, non_blocking=True)  # 非阻塞传输
        
        inp = batch['inp']
        gt = batch['gt']
        
        # 设置输入
        if 'task_id' in batch:
            task_ids = batch['task_id']
            model.set_input(inp, gt, task_ids)
        else:
            model.set_input(inp, gt)
        
        # 优化参数（包括前向、反向传播和参数更新）
        model.optimize_parameters()
        loss = model.loss_G
        
        # 收集损失用于统计
        batch_loss = [
            torch.zeros_like(loss) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(batch_loss, loss)
        loss_list.extend(batch_loss)
        
        # 内存清理 - 删除不需要的变量
        del inp, gt, batch_loss
        if 'task_id' in batch:
            del task_ids
        del batch
        
        # 定期清理CUDA缓存
        if idx % 50 == 0:  # 每50个batch清理一次
            torch.cuda.empty_cache()
        
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()
    
    # 训练结束后清理内存
    torch.cuda.empty_cache()
    
    loss = [i.item() for i in loss_list]
    return mean(loss)


def print_model_parameters(model):
    table = PrettyTable(['Layer Name', 'Parameters Shape', 'Requires Grad'])
    table.align['Layer Name'] = 'l'
    table.align['Parameters Shape'] = 'l'
    table.align['Requires Grad'] = 'c'
    
    for name, param in model.named_parameters():
        table.add_row([name, str(list(param.shape)), str(param.requires_grad)])
    
    if local_rank == 0:
        log('\nModel Parameters:')
        log(str(table))
        
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

    model = model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    model = model.module

    sam_checkpoint = torch.load(config['sam_checkpoint'])

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
            log("These unmatched layers will be randomly initialized and trained from scratch")

    load_filtered_state_dict(model, sam_checkpoint)

    for name, para in model.named_parameters():
        if "image_encoder" in name and "prompt_generator" not in name:
            para.requires_grad_(False)
        if "image_encoder" in name and "Adapter" in name:
            para.requires_grad_(True)
        if "image_encoder" in name and "experts" in name:
            para.requires_grad_(True)
        if "image_encoder" in name and "gate" in name:
            para.requires_grad_(True)
        
        # 检查是否是不匹配的权重层，如果是则设置为可训练
        model_dict = model.state_dict()
        sam_checkpoint_keys = set(sam_checkpoint.keys())
        if name in sam_checkpoint_keys:
            # 检查形状是否匹配
            if name in model_dict and model_dict[name].shape != sam_checkpoint[name].shape:
                para.requires_grad_(True)
                if local_rank == 0:
                    log(f"Setting unmatched layer to trainable: {name} "
                        f"(model: {model_dict[name].shape} vs checkpoint: {sam_checkpoint[name].shape})")
        elif any(keyword in name for keyword in ["mask_decoder", "prompt_encoder", "classifier", "head"]):
            # 对于常见的输出层，即使不在checkpoint中也设为可训练
            para.requires_grad_(True)
            if local_rank == 0:
                log(f"Setting output layer to trainable: {name}")

    print_model_parameters(model)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

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
    
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        train_loss_G = train(train_loader, model, scaler)  # 传入scaler
        lr_scheduler.step()

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

            save(config, model, save_path, 'last', optimizer, lr_scheduler, epoch, scaler)

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

            if local_rank == 0:
                save(config, model, save_path, str(epoch), optimizer, lr_scheduler, epoch, scaler)

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        save(config, model, save_path, 'best', optimizer, lr_scheduler, epoch, scaler)
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        save(config, model, save_path, 'best', optimizer, lr_scheduler, epoch, scaler)

                t = timer.t()
                prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
                t_epoch = utils.time_text(t - t_epoch_start)
                t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
                log_info.append(
                    'epoch train + val time: {} {}/{}'.format(t_epoch, t_elapsed, t_all)
                )

                # 处理多任务结果显示
                if isinstance(seg_eval_table, list):
                    # 多任务：显示所有任务的表格
                    for i, table in enumerate(seg_eval_table):
                        log_info.append(f'\n=== {table.title if hasattr(table, "title") else f"Task {i}"} ===')
                        log_info.append(str(table))
                        if i < len(normed_confusionMatrix):
                            log_info.append(f'Confusion Matrix for Task {i}:')
                            log_info.append(str(normed_confusionMatrix[i]))
                else:
                    # 单任务：显示单个表格
                    log_info.append(str(seg_eval_table))
                    log_info.append('Confusion Matrix:')
                    log_info.append(str(normed_confusionMatrix))

                log('\n'.join(log_info))
                writer.flush()


def save(config, model, save_path, name, optimizer=None, lr_scheduler=None, epoch=None, scaler=None):
    """保存完整的训练状态，支持完全恢复训练"""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    
    # 保存优化器状态
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    # 保存学习率调度器状态
    if lr_scheduler is not None:
        checkpoint['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
    
    # 保存当前epoch
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    # 保存混合精度缩放器状态
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # 保存随机数状态
    checkpoint['random_states'] = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    
    save_file = os.path.join(save_path, f"model_epoch_{name}.pth")
    
    if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
        if config['model']['args']['encoder_mode']['name'] == 'evp':
            # 特殊处理：只保存特定层
            prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
            decode_head = model.encoder.decode_head.state_dict()
            checkpoint.update({
                "prompt": prompt_generator, 
                "decode_head": decode_head
            })
            torch.save(checkpoint, save_file.replace('.pth', '_prompt.pth'))
        else:
            torch.save(checkpoint, save_file)
    else:
        torch.save(checkpoint, save_file)


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