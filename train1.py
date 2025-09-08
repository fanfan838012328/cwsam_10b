import argparse
import os
import math

import yaml
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

import datasets
import models
import utils
from statistics import mean
import torch
import torch.distributed as dist
import gc  # 添加垃圾回收模块
import psutil  # 添加系统内存监控模块

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

# Hopper(H20) 优化：开启 TF32 / SDPA 并启用 cudnn benchmark
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True)
    except Exception:
        pass
except Exception:
    pass


def print_memory_usage(stage=""):
    """打印当前内存使用情况"""
    if local_rank == 0:
        # CPU内存使用情况
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
        system_memory = psutil.virtual_memory()
        
        # GPU内存使用情况
        if torch.cuda.is_available():
            gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024  # GB
            gpu_memory_cached = torch.cuda.memory_reserved() / 1024 / 1024 / 1024  # GB
            print(f"[{stage}] CPU内存: {cpu_memory:.2f}GB, 系统内存使用率: {system_memory.percent:.1f}%")
            print(f"[{stage}] GPU{local_rank}内存: 分配{gpu_memory_allocated:.2f}GB, 缓存{gpu_memory_cached:.2f}GB")
        else:
            print(f"[{stage}] CPU内存: {cpu_memory:.2f}GB, 系统内存使用率: {system_memory.percent:.1f}%")


def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_training_health(model, optimizer, epoch, loss_history):
    """全面检查训练状态健康度"""
    issues_found = []
    
    # 检查模型参数是否包含NaN/Inf
    for name, param in model.named_parameters():
        if param.requires_grad:
            if torch.isnan(param).any():
                issues_found.append(f"参数 {name} 包含NaN")
            if torch.isinf(param).any():
                issues_found.append(f"参数 {name} 包含Inf")
            
            # 检查参数范围是否异常
            param_max = param.abs().max().item()
            if param_max > 100:
                issues_found.append(f"参数 {name} 数值过大: {param_max:.2f}")
    
    # 检查学习率
    current_lr = optimizer.param_groups[0]['lr']
    if current_lr > 1e-2:
        issues_found.append(f"学习率可能过高: {current_lr:.6f}")
    elif current_lr < 1e-8:
        issues_found.append(f"学习率可能过低: {current_lr:.6f}")
    
    # 检查损失历史
    if len(loss_history) >= 3:
        recent_losses = loss_history[-3:]
        if all(l > 10.0 for l in recent_losses):
            issues_found.append("连续3个epoch损失都很高")
        elif any(math.isnan(l) or math.isinf(l) for l in recent_losses):
            issues_found.append("最近损失包含异常值")
    
    if issues_found and local_rank == 0:
        log(f"训练健康检查 Epoch {epoch} 发现问题:")
        for issue in issues_found:
            log(f"  - {issue}")
    
    return len(issues_found) == 0


def check_and_adjust_learning_rate(optimizer, epoch, loss_history, patience=3):
    """检查并调整学习率以防止训练不稳定"""
    current_lr = optimizer.param_groups[0]['lr']
    
    # 如果学习率过高且最近损失不稳定，降低学习率
    if len(loss_history) >= patience:
        recent_losses = loss_history[-patience:]
        if any(l > 10.0 for l in recent_losses):  # 损失过高
            new_lr = current_lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            if local_rank == 0:
                log(f"检测到高损失，将学习率从 {current_lr:.6f} 降低到 {new_lr:.6f}")
            return True
    
    # 检查是否需要预热学习率（前几个epoch使用较小的学习率）
    if epoch <= 3:
        warmup_factor = 0.1
        adjusted_lr = current_lr * warmup_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr
        if local_rank == 0:
            log(f"预热阶段，将学习率调整为 {adjusted_lr:.6f}")
        return True
    
    return False


def save_checkpoint_optimized(model, optimizer, save_path, epoch_or_name):
    """优化的检查点保存函数，减少内存占用"""
    try:
        # 获取模型状态，但不立即创建完整的checkpoint字典
        model_state = model.state_dict()
        
        # 逐步构建checkpoint，避免峰值内存占用
        checkpoint_path = os.path.join(save_path, f"model_epoch_{epoch_or_name}.pth")
        
        # 直接保存，避免中间变量
        torch.save({
            'model': model_state,
            'optimizer': optimizer.state_dict() if optimizer else None,
        }, checkpoint_path)
        
        # 立即清理
        del model_state
        cleanup_memory()
        
        if local_rank == 0:
            log(f"检查点已保存: {checkpoint_path}")
            
    except Exception as e:
        if local_rank == 0:
            log(f"保存检查点时出错: {e}")
        cleanup_memory()


def load_checkpoint_optimized(model, checkpoint_path, device_id):
    """优化的检查点加载函数，减少内存占用"""
    try:
        if local_rank == 0:
            log(f'加载checkpoint: {checkpoint_path}')
            
        # 直接加载到指定GPU，避免CPU内存占用
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{device_id}')
        
        # 提取模型状态并立即加载
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
            if local_rank == 0:
                log('模型权重加载完成')
        
        # 立即删除checkpoint以释放内存
        del checkpoint
        cleanup_memory()
        
        return True
        
    except Exception as e:
        if local_rank == 0:
            log(f"加载检查点时出错: {e}")
        cleanup_memory()
        return False


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
            if hasattr(v, 'shape'):
                log('  {}: shape={}'.format(k, tuple(v.shape)))
            else:
                log('  {}: {}'.format(k, v))

    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=8,  # 减少worker数量，降低内存占用
        pin_memory=True,  # 启用pin_memory加速GPU传输
        sampler=sampler,
        prefetch_factor=2,  # 减少预取数据量
        persistent_workers=True,  # 重用worker进程
    )
    return loader


def multi_task_collate_fn(batch):
    """自定义collate函数，处理不同类别数量的多任务数据"""
    # 按任务ID分组
    task_groups = {}
    for item in batch:
        task_id = item['task_id'].item() if hasattr(item['task_id'], 'item') else item['task_id']
        if task_id not in task_groups:
            task_groups[task_id] = []
        task_groups[task_id].append(item)
    
    # 找到最大的类别数，用于padding
    max_classes = 0
    for task_id, task_items in task_groups.items():
        num_classes = task_items[0]['gt'].shape[0]
        max_classes = max(max_classes, num_classes)
    
    # 为每个任务创建独立的batch并padding到相同大小
    batched_data = {'inp': [], 'gt': [], 'task_id': []}
    
    for task_id, task_items in task_groups.items():
        # 堆叠同一任务的数据
        task_inp = torch.stack([item['inp'] for item in task_items])
        task_gt = torch.stack([item['gt'] for item in task_items])
        task_ids = torch.tensor([task_id] * len(task_items))
        
        # 如果当前任务的类别数少于最大值，进行padding
        current_classes = task_gt.shape[1]
        if current_classes < max_classes:
            padding_size = max_classes - current_classes
            padding = torch.zeros(task_gt.shape[0], padding_size, task_gt.shape[2], task_gt.shape[3])
            task_gt = torch.cat([task_gt, padding], dim=1)
        
        batched_data['inp'].append(task_inp)
        batched_data['gt'].append(task_gt)
        batched_data['task_id'].append(task_ids)
    
    # 将不同任务的数据连接起来
    final_inp = torch.cat(batched_data['inp'], dim=0)
    final_gt = torch.cat(batched_data['gt'], dim=0)
    final_task_ids = torch.cat(batched_data['task_id'], dim=0)
    
    return {
        'inp': final_inp,
        'gt': final_gt,
        'task_id': final_task_ids
    }


def make_multi_data_loader(datasets_config, tag=''):
    """创建多任务数据加载器"""
    if datasets_config is None:
        return None
    
    all_datasets = []
    total_size = 0
    batch_size = 1  # 默认batch_size
    
    for dataset_config in datasets_config:
        dataset = datasets.make(dataset_config['dataset'])
        dataset = datasets.make(dataset_config['wrapper'], args={'dataset': dataset})
        all_datasets.append(dataset)
        total_size += len(dataset)
        
        # 获取batch_size（如果配置了的话）
        if 'batch_size' in dataset_config:
            batch_size = dataset_config['batch_size']
        
        if local_rank == 0:
            log('{} dataset (task_id={}): size={}'.format(
                tag, dataset_config['dataset']['args'].get('task_id', 'unknown'), len(dataset)))
            
            # 显示第一个样本的信息
            sample = dataset[0]
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    log('  {}: shape={}'.format(k, tuple(v.shape)))
                else:
                    log('  {}: {}'.format(k, v))
    
    # 合并多个数据集
    combined_dataset = torch.utils.data.ConcatDataset(all_datasets)
    
    if local_rank == 0:
        log('{} combined dataset: total_size={}, batch_size={}'.format(tag, total_size, batch_size))
    
    sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset)
    loader = DataLoader(
        combined_dataset, 
        batch_size=batch_size,  # 使用可配置的batch_size
        shuffle=False, 
        num_workers=2,  # 减少worker数量，降低内存占用
        pin_memory=True,  # 启用pin_memory加速GPU传输
        sampler=sampler,
        collate_fn=multi_task_collate_fn,  # 使用自定义collate函数
        prefetch_factor=1,  # 减少预取数据量
        persistent_workers=True,  # 重用worker进程
    )
    return loader


def make_data_loaders():
    print_memory_usage("数据加载器创建前")
    
    # 检查是否使用多任务配置
    if 'train_datasets' in config:
        # 多任务配置
        train_loader = make_multi_data_loader(config.get('train_datasets'), tag='train')
    else:
        # 单任务配置（向后兼容）
        train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    
    print_memory_usage("训练数据加载器创建后")
    
    if 'val_datasets' in config:
        # 多任务验证配置
        val_loader = make_multi_data_loader(config.get('val_datasets'), tag='val')
    else:
        # 单任务验证配置（向后兼容）
        val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    
    print_memory_usage("验证数据加载器创建后")
    cleanup_memory()
    
    return train_loader, val_loader

def eval_psnr(loader, model, config):
    model.eval()
    eval_type = config.get('eval_type')

    # 仅在rank 0上初始化tqdm，以避免多行进度条
    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc=f'eval on rank {local_rank}')
    else:
        pbar = None

    # 所有rank都需要准备自己的metric averager
    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()

    # 每个rank都需要初始化自己的SegmentationMetric
    if eval_type == 'seg':
        if 'task_configs' in config['model']['args']:
            task_configs = config['model']['args']['task_configs']
            class_num = max(task_config['num_classes'] for task_config in task_configs)
        else:
            class_num = config['model']['args']['num_classes']

        if 'val_datasets' in config:
            ignore_background = config['val_datasets'][0]['dataset']['args']['ignore_bg']
        else:
            ignore_background = config['val_dataset']['dataset']['args']['ignore_bg']

        metric_seg = SegmentationMetric(class_num, ignore_background)

    # 数据处理循环
    for i, batch in enumerate(loader):
        try:
            # 打印当前处理的batch索引和文件名（如果存在）
            filename_info = ""
            if 'filename' in batch:
                filename_info = f", filename: {batch['filename'][0]}" # 只打印batch中第一个文件名
            
            # log(f"[Rank {local_rank}] Processing batch {i}/{len(loader)}{filename_info}")

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            inp = batch['inp']

            with torch.no_grad():
                if 'task_id' in batch:
                    task_id = batch['task_id'][0].item()
                    output_masks = model.infer(inp, task_id=task_id)
                else:
                    output_masks = model.infer(inp)
                
                # 多类分割使用 softmax 概率
                pred = torch.softmax(output_masks, dim=1)

            # metric_fn的计算可以保留，因为Averager是非分布式聚合的
            if eval_type != 'seg':
                 metric_fn = utils.calc_f1 # 根据eval_type获取metric_fn
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
                    metric_seg.addBatch(mask_index, gt_index)

            if pbar:
                pbar.update(1)

        except Exception as e:
            log(f"[Rank {local_rank}] ERROR at batch {i}{filename_info}: {e}")
            import traceback
            log(traceback.format_exc())
            # 出错时也需要同步，否则其他rank会卡住
            dist.barrier()
            continue # 继续处理下一个batch
    
    if pbar:
        pbar.close()

    # 手动同步所有进程，确保所有进程都完成了计算
    log(f"[Rank {local_rank}] Finished processing all batches. Waiting at barrier.")
    dist.barrier()
    log(f"[Rank {local_rank}] Passed barrier.")

    # 收集所有GPU的结果到Rank 0
    if eval_type == 'seg':
        # 将混淆矩阵从numpy转为tensor以进行分布式收集
        confusion_matrix_tensor = torch.tensor(metric_seg.confusionMatrix, device=device)
        
        # 创建一个列表来接收所有rank的混淆矩阵
        all_confusion_matrices = [torch.zeros_like(confusion_matrix_tensor) for _ in range(dist.get_world_size())]
        
        # 使用all_gather来收集
        dist.all_gather(all_confusion_matrices, confusion_matrix_tensor)
        
        # Rank 0 聚合结果
        if local_rank == 0:
            final_confusion_matrix = torch.stack(all_confusion_matrices).sum(dim=0)
            # 将聚合后的tensor转回numpy，并更新到metric_seg对象中
            metric_seg.confusionMatrix = final_confusion_matrix.cpu().numpy()
            log("Successfully gathered and aggregated confusion matrices from all ranks.")
        else:
             # 非rank 0的进程不需要做后续计算
             return 0, 0, 0, 0, 'none', 'none', 'none', 'none', None, None

    # 只有Rank 0 计算最终指标并打印
    if local_rank == 0:
        # 这里的 averager 结果仅为 rank 0 的局部结果，如果需要全局指标需要额外同步
        val_metric1_avg = val_metric1.item()
        val_metric2_avg = val_metric2.item()
        val_metric3_avg = val_metric3.item()
        val_metric4_avg = val_metric4.item()
        
        table = None
        normed_confusionMatrix = None
        if eval_type == 'seg':
            # ... (此处省略了之前详细的mIoU, F1等计算和表格生成代码)
            # ... 你需要将之前的表格生成逻辑放在这里
            # 确保使用聚合后的 metric_seg 对象进行计算
            log("Rank 0 is now calculating final metrics.")
            try:
                oa = metric_seg.overallAccuracy()
                oa = np.nan_to_num(oa, nan=0.0, posinf=0.0, neginf=0.0)
                oa = np.around(oa, decimals=4)
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in OA calculation: {e}")
                oa = 0.0
            
            try:
                mIoU, IoU = metric_seg.meanIntersectionOverUnion()
                mIoU = np.nan_to_num(mIoU, nan=0.0, posinf=0.0, neginf=0.0)
                IoU = np.nan_to_num(IoU, nan=0.0, posinf=0.0, neginf=0.0)
                mIoU = np.around(mIoU, decimals=4)
                IoU = np.around(IoU, decimals=4)
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in IoU calculation: {e}")
                mIoU = 0.0
                IoU = np.zeros(class_num)
            
            # 处理可能出现的除零情况 - 更安全的版本
            try:
                confusion_sum_axis0 = metric_seg.confusionMatrix.sum(axis=0)
                confusion_sum_axis0 = np.where(confusion_sum_axis0 == 0, 1e-10, confusion_sum_axis0)
                p = np.diag(metric_seg.confusionMatrix) / confusion_sum_axis0
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                p = np.around(p, decimals=4)
                mp = np.nanmean(p) if len(p) > 0 else 0.0
                mp = np.around(mp, decimals=4)
                
                confusion_sum_axis1 = metric_seg.confusionMatrix.sum(axis=1)
                confusion_sum_axis1 = np.where(confusion_sum_axis1 == 0, 1e-10, confusion_sum_axis1)
                r = np.diag(metric_seg.confusionMatrix) / confusion_sum_axis1
                r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
                r = np.around(r, decimals=4)
                mr = np.nanmean(r) if len(r) > 0 else 0.0
                mr = np.around(mr, decimals=4)
                
                # 处理F1计算中的除零情况 - 更安全
                f1 = np.zeros_like(p)
                denominator = p + r
                valid_mask = denominator > 1e-10
                f1[valid_mask] = (2 * p[valid_mask] * r[valid_mask]) / denominator[valid_mask]
                f1 = np.nan_to_num(f1, nan=0.0, posinf=0.0, neginf=0.0)
                f1 = np.around(f1, decimals=4)
                mf1 = np.nanmean(f1) if len(f1) > 0 else 0.0
                mf1 = np.around(mf1, decimals=4)
                
                # 处理混淆矩阵归一化 - 更安全
                row_sums = metric_seg.confusionMatrix.sum(axis=0)
                row_sums = np.where(row_sums == 0, 1e-10, row_sums)
                normed_confusionMatrix = metric_seg.confusionMatrix / row_sums[np.newaxis, :]
                normed_confusionMatrix = np.nan_to_num(normed_confusionMatrix, nan=0.0, posinf=0.0, neginf=0.0)
                normed_confusionMatrix = np.around(normed_confusionMatrix, decimals=3)
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in metric calculation: {e}")
                # 设置默认值
                p = np.zeros(class_num)
                r = np.zeros(class_num)
                f1 = np.zeros(class_num)
                mp = mr = mf1 = 0.0
                normed_confusionMatrix = np.zeros_like(metric_seg.confusionMatrix, dtype=float)
            
            try:
                fwIOU = metric_seg.Frequency_Weighted_Intersection_over_Union()
                fwIOU = np.nan_to_num(fwIOU, nan=0.0, posinf=0.0, neginf=0.0)
                fwIOU = np.around(fwIOU, decimals=4)
            except Exception as e:
                if local_rank == 0:
                    print(f"Error in fwIOU calculation: {e}")
                fwIOU = 0.0
            
            # 处理多任务和单任务的classes获取
            if 'train_datasets' in config:
                # 多任务配置：使用第一个数据集的classes
                classes_list = config['train_datasets'][0]['dataset']['args']['classes']
            else:
                # 单任务配置：使用原始方式
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
        # 在函数的末尾返回正确数量的值
        metric1, metric2, metric3, metric4 = 'none', 'none', 'none', 'none'
        return val_metric1_avg, val_metric2_avg, val_metric3_avg, val_metric4_avg, metric1, metric2, metric3, metric4, table, normed_confusionMatrix
    
    # 如果不是rank 0, 也需要返回同样数量的占位符
    return 0, 0, 0, 0, 'none', 'none', 'none', 'none', None, None


def prepare_training(save_path):
    print_memory_usage("模型创建前")

    # 将resume状态传递给模型，以便决定是否加载预训练权重
    if 'args' not in config['model']:
        config['model']['args'] = {}
    config['model']['args']['resume'] = config.get('resume')

    epoch_start = 1
    
    # 场景1: 从之前的训练中完全恢复 (加载权重并快进学习率)
    if config.get('resume') is not None:
        epoch_start = config.get('resume') + 1
        work_dir = config.get('work_dir', save_path)
        resume_model_path = os.path.join(
            work_dir, 'model_epoch_' + str(config.get('resume')) + '.pth'
        )
        
        if local_rank == 0:
            log(f'恢复训练，加载 checkpoint: {resume_model_path}')
            
        model_config = config['model'].copy()
        if 'lora_l2_weight' in config:
            model_config['args']['lora_l2_weight'] = config['lora_l2_weight']
        model = models.make(model_config).cuda()
        print_memory_usage("模型直接在GPU上创建后")
        
        success = load_checkpoint_optimized(model, resume_model_path, local_rank)
        if not success:
            if local_rank == 0:
                log("警告: 使用优化加载失败，回退到原始方法")
            checkpoint = torch.load(resume_model_path, map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint['model'], strict=False)
            del checkpoint
            cleanup_memory()
        
        print_memory_usage("checkpoint加载完成后内存清理")
        if local_rank == 0:
            log('从 epoch {} 恢复训练'.format(epoch_start))
            
    # 场景2: 开始新训练，但从指定checkpoint加载权重 (学习率从头开始)
    elif config.get('sam_checkpoint') is not None:
        model_config = config['model'].copy()
        if 'lora_l2_weight' in config:
            model_config['args']['lora_l2_weight'] = config['lora_l2_weight']
        model = models.make(model_config).cuda()
        print_memory_usage("新模型直接在GPU上创建后")
        
        checkpoint_path = config.get('sam_checkpoint')
        if local_rank == 0:
            log(f'加载指定权重进行新训练: {checkpoint_path}')
        
        success = load_checkpoint_optimized(model, checkpoint_path, local_rank)
        if not success:
            if local_rank == 0:
                log("警告: 使用优化加载失败，回退到原始方法")
            checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}')
            model.load_state_dict(checkpoint['model'], strict=False)
            del checkpoint
            cleanup_memory()
            
        print_memory_usage("从 sam_checkpoint 加载权重后")
        
        # 检查加载的权重是否包含异常值
        if local_rank == 0:
            log('检查加载的模型权重健康状态...')
            nan_params = 0
            inf_params = 0
            large_params = 0
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    nan_params += 1
                    log(f"警告: 参数 {name} 包含NaN值")
                if torch.isinf(param).any():
                    inf_params += 1
                    log(f"警告: 参数 {name} 包含Inf值")
                param_max = param.abs().max().item()
                if param_max > 100:
                    large_params += 1
                    log(f"警告: 参数 {name} 数值过大: {param_max:.2f}")
            
            log(f'权重健康检查完成: NaN参数={nan_params}, Inf参数={inf_params}, 过大参数={large_params}')
            log('权重加载完成，将从 epoch 1 开始新的训练')
        # epoch_start 保持为 1, 学习率调度器将从头开始

    # 场景3: 完全从头开始训练
    else:
        # 不恢复训练的情况，直接在GPU上创建模型
        model_config = config['model'].copy()
        if 'lora_l2_weight' in config:
            model_config['args']['lora_l2_weight'] = config['lora_l2_weight']
        model = models.make(model_config).cuda()
        print_memory_usage("新模型直接在GPU上创建后")
    
    # 将模型设置为channels_last内存格式以提高性能
    model = model.to(memory_format=torch.channels_last)
    print_memory_usage("模型转换为channels_last后")

    # DDP包装
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=False,
    )
    print_memory_usage("DDP模型创建后")

    if local_rank == 0:
        log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    
    return model.module, epoch_start


def lora_l2_loss(model, alpha=0.001):
    """计算LoRA权重的L2正则化损失"""
    l2_loss = 0
    for name, param in model.named_parameters():
        if 'lora' in name.lower() and 'linear_a' in name:
            l2_loss += torch.norm(param, p=2)
    return alpha * l2_loss


def apply_warmup_lr(optimizer, epoch, config):
    """应用学习率warmup策略"""
    if 'warmup_epochs' not in config:
        return None
        
    warmup_epochs = config['warmup_epochs']
    warmup_factor = config.get('warmup_factor', 0.1)
    
    if epoch < warmup_epochs:
        base_lr = config['optimizer']['args']['lr']
        progress = (epoch + 1) / warmup_epochs
        current_lr = base_lr * (warmup_factor + (1 - warmup_factor) * progress)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
            
        if local_rank == 0:
            log(f"Warmup epoch {epoch+1}/{warmup_epochs}, LR: {current_lr:.6f}")
        return current_lr
    return None


def train(train_loader, model, scheduler, scaler):
    model.train()

    if local_rank == 0:
        pbar = tqdm(total=len(train_loader), leave=False, desc='train')
    else:
        pbar = None

    loss_list = []
    nan_count = 0
    # 关闭梯度裁剪，让模型自由更新参数
    # 使用bfloat16和更低学习率来保证数值稳定性
    max_grad_norm = None  # 不再使用梯度裁剪
    
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            # channels_last + 非阻塞拷贝
            if isinstance(v, torch.Tensor):
                v = v.to(device, non_blocking=True)
                if v.dim() == 4:
                    v = v.to(memory_format=torch.channels_last)
            batch[k] = v
        inp = batch['inp']
        gt = batch['gt']
        
        # 检查是否包含task_id
        if 'task_id' in batch:
            task_ids = batch['task_id']
            model.set_input(inp, gt, task_ids)
        else:
            # 单任务模式，向后兼容
            model.set_input(inp, gt)
        
        # 改用bfloat16以提高数值稳定性，特别适合大参数模型
        try:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                # 手动执行优化步骤以增加控制
                model.forward()
                model.optimizer.zero_grad(set_to_none=True) # 使用 set_to_none=True 进一步优化内存
                
                # 检查前向传播结果
                if torch.isnan(model.pred_mask).any() or torch.isinf(model.pred_mask).any():
                    if local_rank == 0:
                        log(f"警告: batch {batch_idx} 前向传播产生了NaN/Inf，跳过此batch")
                    cleanup_memory() # 出现异常时清理内存
                    continue
                
                # 计算损失
                model.backward_G()
                
            # 在 GradScaler 外部计算损失，因为它内部会进行类型转换
            # 但反向传播需要通过 scaler 完成
            if torch.isnan(model.loss_G) or torch.isinf(model.loss_G):
                nan_count += 1
                if local_rank == 0:
                    log(f"警告: batch {batch_idx} 损失为NaN/Inf: {model.loss_G.item()}")
                # 跳过这个batch，不更新参数
                cleanup_memory() # 出现异常时清理内存
                continue
            
            # 使用 GradScaler 缩放损失并反向传播
            scaler.scale(model.loss_G).backward()
            
            # 在梯度裁剪前取消缩放
            scaler.unscale_(model.optimizer)
            
            # 关闭梯度裁剪，只计算梯度范数用于监控
            total_grad_norm_tensor = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.parameters() if p.grad is not None]))
            total_grad_norm = total_grad_norm_tensor.item()

            # 只在梯度范数真正异常时才警告（减少日志噪音）
            if torch.isinf(total_grad_norm_tensor) or torch.isnan(total_grad_norm_tensor):
                if local_rank == 0:
                    log(f"严重警告: batch {batch_idx} 梯度包含 NaN/Inf，已跳过更新")
            elif total_grad_norm > 10.0:  # 监控大梯度但不裁剪
                if local_rank == 0 and batch_idx % 100 == 0:  # 每100个batch最多警告一次
                    log(f"梯度范数较大: {total_grad_norm:.2f} (仅监控，不裁剪)")

            # scaler.step() 会自动检查梯度是否为NaN/Inf，并决定是否更新
            scaler.step(model.optimizer)
            
            # 更新缩放器，为下一次迭代做准备
            scaler.update()
            
            scheduler.step()
                
        except Exception as e:
            if local_rank == 0:
                import traceback
                log(f"训练异常 batch {batch_idx}: {e}")
                log(traceback.format_exc())
            cleanup_memory() # 出现异常时清理内存
            continue
        
        # 收集损失
        batch_loss = [
            torch.zeros_like(model.loss_G) for _ in range(dist.get_world_size())
        ]
        dist.all_gather(batch_loss, model.loss_G)
        loss_list.extend(batch_loss)
        
        if pbar is not None:
            # 显示当前损失和累计NaN数量
            current_loss = model.loss_G.item()
            pbar.set_postfix(
                {
                    'loss': f'{current_loss:.4f}',
                    'nan_count': nan_count,
                    'grad_norm': f'{total_grad_norm:.2f}'
                    if 'total_grad_norm' in locals()
                    else 'N/A',
                    'scale': scaler.get_scale(),
                }
            )
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # 统计信息
    if local_rank == 0 and nan_count > 0:
        log(f"训练完成，总共跳过 {nan_count} 个NaN/Inf batch")

    if not loss_list:
        if local_rank == 0:
            log("警告: 整个 epoch 中没有任何一个 batch 成功计算损失，返回 NaN。")
        return float('nan')

    loss = [i.item() for i in loss_list]
    return mean(loss)


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

    # 1. 准备模型和训练起点（使用优化的内存管理）
    model, epoch_start = prepare_training(save_path)
    print_memory_usage("模型准备完成后")

    # 2. 精确控制可训练参数
    for param in model.parameters():
        param.requires_grad = False

    # 启用 LoRA 参数
    try:
        from models.lora import LoRA as LoRAModule
    except Exception:
        LoRAModule = None

    if LoRAModule is not None:
        for module in model.modules():
            if isinstance(module, LoRAModule):
                for name, p in module.named_parameters():
                    if name.startswith('qkv.'):
                        p.requires_grad = False
                    else:
                        p.requires_grad = True

    if hasattr(model, 'image_encoder'):
        if hasattr(model.image_encoder, 'w_a'):
            for lin in model.image_encoder.w_a:
                for p in lin.parameters():
                    p.requires_grad = True
        if hasattr(model.image_encoder, 'w_b'):
            for lin in model.image_encoder.w_b:
                for p in lin.parameters():
                    p.requires_grad = True

    # 启用MoE参数（如果模型使用了MoE）
    if hasattr(model, 'image_encoder'):
        # 检查是否是MoE模型
        if hasattr(model.image_encoder, 'moe_modules') and hasattr(model.image_encoder, 'get_moe_parameters'):
            print("检测到MoE模型，启用MoE专家参数...")
            moe_params = model.image_encoder.get_moe_parameters()
            for param in moe_params:
                param.requires_grad = True
            
            trainable_moe_params = sum(p.numel() for p in moe_params if p.requires_grad)
            print(f"已启用 {trainable_moe_params/1e6:.1f}M MoE参数进行训练")
    
    # 启用 projection 和 解码头
    for name, p in model.named_parameters():
        if (
            name.startswith('projection')
            or name.startswith('mask_decoder')
            or '.mask_decoders.' in name
            or name.startswith('no_mask_embed')
            or name.startswith('pe_layer')
            or 'adapter' in name
        ):
            p.requires_grad = True

    cleanup_memory()
    print_memory_usage("参数设置完成后")

    # 3. 在参数冻结后，只为可训练的参数创建优化器
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = utils.make_optimizer(trainable_params, config['optimizer'])
    model.optimizer = optimizer
    print_memory_usage("优化器创建后")
    
    # 4. 创建学习率调度器
    # 对于从检查点加载的训练，使用更温和的调度器
    if config.get('sam_checkpoint') is not None:
        # 从检查点开始训练：使用非常小的学习率和温和的余弦退火
        adjusted_lr = config['optimizer']['args']['lr'] * 0.05  # 降低到1/20 (平衡稳定性和收敛速度)
        lr_scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=config['epoch_max'] * len(train_loader),
            eta_min=1e-7  # 调整最小学习率以匹配新的学习率范围
        )
        # 手动设置较低的初始学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = adjusted_lr
        if local_rank == 0:
            log(f"从检查点开始训练，学习率调整为: {adjusted_lr:.2e} (平衡稳定性与收敛速度)")
    else:
        # 从头开始训练：使用原来的 OneCycleLR
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=config['optimizer']['args']['lr'],
            epochs=config['epoch_max'],
            steps_per_epoch=len(train_loader)
        )

    # 如果恢复训练，更新调度器状态
    if config.get('resume') is not None:
        # 对于按批次更新的 OneCycleLR，需要将调度器推进到正确的步数
        steps_to_advance = (epoch_start - 1) * len(train_loader)
        for _ in range(steps_to_advance):
            lr_scheduler.step()
        if local_rank == 0:
            log(f"学习率调度器已更新至 epoch {epoch_start - 1}. 下一个LR为 {lr_scheduler.get_last_lr()[0]}")

    print_memory_usage("最终模型设置完成")

    if local_rank == 0:
        log("\n--- 可训练参数 ---")
        for name, para in model.named_parameters():
            if para.requires_grad:
                log(name)
        log("---------------------------\n")

    print_model_parameters(model)
    if local_rank == 0:
        model_total_params = sum(p.numel() for p in model.parameters())
        model_grad_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
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
    
    # 为混合精度训练初始化 GradScaler (修正了FutureWarning)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    
    # 添加损失历史记录用于训练稳定性监控
    loss_history = []
    
    for epoch in range(epoch_start, epoch_max + 1):
        train_loader.sampler.set_epoch(epoch)
        t_epoch_start = timer.t()
        
        # 应用warmup策略
        warmup_lr = apply_warmup_lr(optimizer, epoch, config)
        
        # 在训练前检查学习率 (OneCycleLR 自动处理预热和调度)
        # check_and_adjust_learning_rate(optimizer, epoch, loss_history)
        
        # 训练前进行健康检查
        is_healthy = check_training_health(model, optimizer, epoch, loss_history)
        if not is_healthy and local_rank == 0:
            log(f"警告: Epoch {epoch} 训练前发现健康问题，将谨慎进行训练")
        
        train_loss_G = train(train_loader, model, lr_scheduler, scaler)
        
        # 检查训练损失是否异常
        if math.isnan(train_loss_G) or math.isinf(train_loss_G):
            if local_rank == 0:
                log(f"错误: Epoch {epoch} 训练损失为 {train_loss_G}，训练不稳定！")
                # 注意: OneCycleLR 会自动管理学习率，通常不需要手动干预
        else:
            # 记录损失历史
            loss_history.append(train_loss_G)
            # 保持历史记录在合理长度
            if len(loss_history) > 20:
                loss_history.pop(0)
        
        # 清理GPU内存
        cleanup_memory()

        if local_rank == 0:
            log_info = [
                '\n ############################ epoch {}/{} ############################'.format(
                    epoch, epoch_max
                )
            ]
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
            
            # 改进损失显示
            if math.isnan(train_loss_G) or math.isinf(train_loss_G):
                log_info.append('train G: loss={} (异常值!)'.format(train_loss_G))
            else:
                log_info.append('train G: loss={:.4f}'.format(train_loss_G))
                
            writer.add_scalars('loss', {'train G': train_loss_G}, epoch)

            # 只有在损失正常时才保存检查点
            if not (math.isnan(train_loss_G) or math.isinf(train_loss_G)):
                save_checkpoint_optimized(model, optimizer, save_path, epoch)
            else:
                log("跳过异常epoch的检查点保存")


        if (epoch_val is not None) and (epoch % epoch_val == 0):
            # 验证之前清理内存
            cleanup_memory()
            print_memory_usage("验证前内存状态")

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
                # save(config, model, save_path, str(epoch))

                if config['eval_type'] != 'ber':
                    if result1 > max_val_v:
                        max_val_v = result1
                        # 使用优化的检查点保存函数保存最佳模型
                        save_checkpoint_optimized(model, optimizer, save_path, "best")
                else:
                    if result3 < max_val_v:
                        max_val_v = result3
                        # 使用优化的检查点保存函数保存最佳模型
                        save_checkpoint_optimized(model, optimizer, save_path, "best")

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
                
                # 关键：显式删除只在 rank 0 上创建的大型变量，帮助垃圾回收
                del (
                    result1, result2, result3, result4,
                    metric1, metric2, metric3, metric4,
                    seg_eval_table, normed_confusionMatrix
                )
                cleanup_memory() # 再次调用内存清理


def save(config, model, save_path, name):
    # 这个独立的save函数现在可以被废弃，因为保存逻辑已经集成到训练循环中
    # 为了安全起见，暂时保留但注释掉内容
    pass
    # if config['model']['name'] == 'segformer' or config['model']['name'] == 'setr':
    #     if config['model']['args']['encoder_mode']['name'] == 'evp':
    #         prompt_generator = model.encoder.backbone.prompt_generator.state_dict()
    #         decode_head = model.encoder.decode_head.state_dict()
    #         torch.save(
    #             {"prompt": prompt_generator, "decode_head": decode_head},
    #             os.path.join(save_path, f"prompt_epoch_{name}.pth"),
    #         )
    #     else:
    #         torch.save(
    #             model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")
    #         )
    # else:
    #     torch.save(
    #         model.state_dict(), os.path.join(save_path, f"model_epoch_{name}.pth")
    #     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default="configs/train/setr/train_setr_evp_cod.yaml"
    )
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    parser.add_argument('--resume', type=int, default=None, help='Resume training from specific epoch')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        if local_rank == 0:
            print('config loaded.')

    # 设置resume参数（如果通过命令行指定）
    if args.resume is not None:
        config['resume'] = args.resume
        if local_rank == 0:
            print(f'Will resume training from epoch {args.resume}')

    save_name = args.name
    if save_name is None:
        save_name = args.config.split('/')[-1][: -len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)

    main(config, save_path, args=args)
