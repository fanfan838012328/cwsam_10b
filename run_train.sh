#!/bin/bash

# 训练不同规模CWSAM模型的脚本

# 设置GPU数量
NUM_GPUS=4

# 默认参数
MODEL_SIZE="3B"
BATCH_SIZE=4
EPOCHS=100
LR=0.00001
TAG=""
CHECKPOINT="./pretrained/sam_vit_h_4b8939.pth"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_size)
      MODEL_SIZE="$2"
      shift
      shift
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --lr)
      LR="$2"
      shift
      shift
      ;;
    --tag)
      TAG="$2"
      shift
      shift
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift
      shift
      ;;
    --gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 根据模型规模调整批次大小
if [ "$MODEL_SIZE" = "1.5B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=8
  fi
elif [ "$MODEL_SIZE" = "2B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=6
  fi
elif [ "$MODEL_SIZE" = "3B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=4
  fi
elif [ "$MODEL_SIZE" = "5B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=2
  fi
elif [ "$MODEL_SIZE" = "7B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=1
  fi
elif [ "$MODEL_SIZE" = "10B" ]; then
  if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=1
  fi
fi

echo "开始训练 CWSAM ${MODEL_SIZE} 模型"
echo "批次大小: ${BATCH_SIZE}"
echo "训练轮数: ${EPOCHS}"
echo "学习率: ${LR}"
echo "使用GPU数量: ${NUM_GPUS}"
echo "预训练权重: ${CHECKPOINT}"

# 构建命令行参数
CMD_ARGS="--model_size ${MODEL_SIZE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} --checkpoint ${CHECKPOINT}"

if [ ! -z "$TAG" ]; then
  CMD_ARGS="${CMD_ARGS} --tag ${TAG}"
fi

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=${NUM_GPUS} train_cwsam.py ${CMD_ARGS}