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

    sampler = torch.utils.data.DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=spec['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True,  # 推荐在多卡训练中开启以加速数据传输
        sampler=sampler,
    )
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader

def eval_psnr(loader, model, config):
    model.eval()
    eval_type = config.get('eval_type')
    class_num = config['model']['args']['num_classes']
    ignore_background = config['val_dataset']['dataset']['args']['ignore_bg']
    
    # 初始化指标计算器
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

    if local_rank == 0:
        pbar = tqdm(total=len(loader), leave=False, desc='val')
    else:
        pbar = None

    for batch in loader:
        for k, v in batch.items():
            batch[k] = v.to(device)

        with torch.no_grad():
            output_masks = model.infer(batch['inp'])
            pred = torch.sigmoid(output_masks)
        
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        val_metric1.add(result1.item(), batch['inp'].shape[0])
        val_metric2.add(result2.item(), batch['inp'].shape[0])
        val_metric3.add(result3.item(), batch['inp'].shape[0])
        val_metric4.add(result4.item(), batch['inp'].shape[0])
        
        if eval_type == 'seg':
            for b in range(pred.shape[0]):
                mask_index = torch.argmax(pred[b], dim=0).cpu().numpy()
                gt_index = torch.argmax(batch['gt'][b], dim=0).cpu().numpy()
                metric_seg.addBatch(mask_index, gt_index)
        
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # --- 分布式聚合 ---
    if eval_type == 'seg':
        confusion_matrix_tensor = torch.from_numpy(metric_seg.confusionMatrix).to(device)
        dist.all_reduce(confusion_matrix_tensor, op=dist.ReduceOp.SUM)
        if local_rank == 0:
            metric_seg.confusionMatrix = confusion_matrix_tensor.cpu().numpy()

    # 聚合 Averager 的数据
    # 假设 Averager 有 .sum 和 .count 属性
    metrics_data = torch.tensor(
        [
            val_metric1.v * val_metric1.n,
            val_metric1.n,
            val_metric2.v * val_metric2.n,
            val_metric2.n,
            val_metric3.v * val_metric3.n,
            val_metric3.n,
            val_metric4.v * val_metric4.n,
            val_metric4.n,
        ]
    ).float().to(device)
    dist.all_reduce(metrics_data, op=dist.ReduceOp.SUM)

    # --- 在主进程计算并返回结果 ---
    if local_rank == 0:
        val_metric1_avg = (metrics_data[0] / metrics_data[1]).item() if metrics_data[1] > 0 else 0
        val_metric2_avg = (metrics_data[2] / metrics_data[3]).item() if metrics_data[3] > 0 else 0
        val_metric3_avg = (metrics_data[4] / metrics_data[5]).item() if metrics_data[5] > 0 else 0
        val_metric4_avg = (metrics_data[6] / metrics_data[7]).item() if metrics_data[7] > 0 else 0
        
        table = None
        normed_confusionMatrix = None
        if eval_type == 'seg':
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
            classes_list = config['train_dataset']['dataset']['args']['classes']
            axis_labels = classes_list[:-1] if ignore_background else classes_list
            title_row = ['metrics', 'average'] + axis_labels
            table = PrettyTable(title_row)
            
            def create_row(name, avg_val, val_list):
                row = [name, avg_val] + val_list.tolist()
                return row[:len(title_row)] + [' '] * (len(title_row) - len(row))

            table.add_row(create_row('IOU', mIoU, IoU))
            table.add_row(create_row('Precision', mp, p))
            table.add_row(create_row('Recall', mr, r))
            table.add_row(create_row('F1', mf1, f1))
            table.add_row(['OA', oa] + [''] * (len(title_row) - 2))
            table.add_row(['FWIOU', fwIOU] + [''] * (len(title_row) - 2))
        
        torch.cuda.empty_cache()
        return val_metric1_avg, val_metric2_avg, val_metric3_avg, val_metric4_avg, metric1, metric2, metric3, metric4, table, normed_confusionMatrix
    else:
        torch.cuda.empty_cache()
        return 0, 0, 0, 0, 'none', 'none', 'none', 'none', None, None


def log_gpu_memory(stage=""):
    """记录GPU显存使用情况"""
    if local_rank == 0 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        log(f"GPU Memory {stage}: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB")

def prepare_training():
    model = models.make(config['model'])
    
    # 将模型移到GPU，但保持fp32参数以支持混合精度训练
    model = model.to(device)
    log_gpu_memory("after model.to(device)")
    
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
    
    # ===== 高效MoE权重初始化逻辑 =====
    def initialize_efficient_moe_from_1_5b(model, sam_checkpoint):
        """
        使用新的高效MoE架构初始化权重
        兼容原始1.5B专家权重
        """
        if local_rank == 0:
            log("开始使用高效MoE架构初始化权重...")
        
        # 首先检查模型是否使用高效MoE
        efficient_moe_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'load_1_5b_expert_weights'):
                efficient_moe_layers.append((name, module))
        
        if len(efficient_moe_layers) == 0:
            if local_rank == 0:
                log("未找到高效MoE层，使用传统初始化方法")
            # 回退到原始初始化方法
            initialize_hierarchical_moe_from_1_5b(model, sam_checkpoint)
            return
        
        if local_rank == 0:
            log(f"找到 {len(efficient_moe_layers)} 个高效MoE层")
        
        # 使用新的权重加载方法
        with torch.no_grad():
            if local_rank == 0:
                log(f"开始加载 {len(efficient_moe_layers)} 个MoE层的权重...")
            
            for layer_name, moe_module in efficient_moe_layers:
                # 使用新的权重加载方法
                moe_module.load_1_5b_expert_weights(sam_checkpoint)
        
        # 统计冻结参数（如果有的话）
        frozen_params = 0
        trainable_params = 0
        for param in model.parameters():
            if param.requires_grad:
                trainable_params += param.numel()
            else:
                frozen_params += param.numel()
        
        if local_rank == 0:
            log(f"高效MoE权重初始化完成")
            log(f"冻结参数: {frozen_params:,}")
            log(f"可训练参数: {trainable_params:,}")
    
    # 调用权重初始化函数
    if config['model']['name'] == 'sam_hierarchical_moe_10b' or 'hierarchical_moe' in config['model']['name']:
        initialize_efficient_moe_from_1_5b(model, sam_checkpoint)

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
