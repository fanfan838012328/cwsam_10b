#!/usr/bin/env python3
import yaml
import argparse

def test_config_reading():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                       default='/mnt/fanfq/project/code/cwsam_10b/configs/multitask/sam_optimized_multitask.yaml', 
                       help='配置文件路径')
    parser.add_argument('--task_id', 
                       type=int,
                       default=0,
                       help='指定推理任务ID')
    args = parser.parse_args()

    # 加载配置
    print(f"读取配置文件: {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 获取任务配置信息
    task_configs = config.get('task_configs', [])
    print(f"顶级task_configs: {len(task_configs)} 项")
    
    # 如果顶级没有task_configs，尝试从model.args中获取
    if not task_configs and 'model' in config and 'args' in config['model']:
        task_configs = config['model']['args'].get('task_configs', [])
        print(f"从model.args获取task_configs: {len(task_configs)} 项")
    
    print(f"找到的task_configs:")
    for i, task_config in enumerate(task_configs):
        print(f"  {i}: task_id={task_config.get('task_id')}, name={task_config.get('name')}, num_classes={task_config.get('num_classes')}")
    
    # 查找指定的任务ID
    current_task_config = None
    for task_config in task_configs:
        if task_config['task_id'] == args.task_id:
            current_task_config = task_config
            break
    
    if current_task_config:
        print(f"\n找到任务ID {args.task_id}: {current_task_config['name']}, 类别数: {current_task_config['num_classes']}")
    else:
        print(f"\n未找到任务ID {args.task_id}")

if __name__ == '__main__':
    test_config_reading()