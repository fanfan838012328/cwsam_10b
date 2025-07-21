#!/bin/bash

# 内存优化的CWSAM训练脚本

# 设置环境变量以优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1

# 默认参数
MODEL_SIZE="1.5B"
NUM_GPUS=4
BATCH_SIZE=1
INPUT_SIZE=512
TAG="memory_opt"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_size)
      MODEL_SIZE="$2"
      shift
      shift
      ;;
    --gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --input_size)
      INPUT_SIZE="$2"
      shift
      shift
      ;;
    --tag)
      TAG="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 根据模型大小选择配置
if [ "$MODEL_SIZE" = "1.5B" ]; then
    CONFIG_FILE="configs/train/sam/train_sam_moe_1dot5b_optimized.yaml"
    if [ -z "$BATCH_SIZE" ]; then
        BATCH_SIZE=1
    fi
elif [ "$MODEL_SIZE" = "3B" ]; then
    CONFIG_FILE="configs/train/sam/train_sam_moe_3b_optimized.yaml"
    if [ -z "$BATCH_SIZE" ]; then
        BATCH_SIZE=1
    fi
else
    echo "目前只支持1.5B和3B模型的内存优化版本"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "配置文件不存在: $CONFIG_FILE"
    echo "请先创建对应的优化配置文件"
    exit 1
fi

echo "开始内存优化训练 CWSAM ${MODEL_SIZE} 模型"
echo "配置文件: ${CONFIG_FILE}"
echo "批次大小: ${BATCH_SIZE}"
echo "输入尺寸: ${INPUT_SIZE}"
echo "使用GPU数量: ${NUM_GPUS}"

# 构建实验名称
EXP_NAME="cwsam_${MODEL_SIZE,,}_${TAG}_bs${BATCH_SIZE}_${INPUT_SIZE}"

# 启动训练
torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} --master_port=29500 \
    train_memory_optimized.py \
    --config ${CONFIG_FILE} \
    --name ${EXP_NAME} \
    --tag ${TAG}

echo "训练完成！"