# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a computer vision research project implementing **ClassWise-SAM**, a specialized version of Meta's Segment Anything Model (SAM) adapted for remote sensing image segmentation. The project focuses on scaling SAM from 1.5B to 10B parameters using hierarchical Mixture of Experts (MoE) architecture for improved land cover classification.

## Key Architecture Components

### Model Architecture
- **Base Model**: SAM (Segment Anything Model) with ViT-Huge backbone
- **Scaling Strategy**: Hierarchical MoE with 6 expert groups × 16 experts per group
- **Target Classes**: 33 land cover classes (agriculture, urban, water bodies, etc.)
- **Input Size**: 512x512 images
- **Model Variants**:
  - `sam_hierarchical_moe_10b`: 10B parameter hierarchical MoE model
  - `sam_moe_3b`: 3B parameter standard MoE model (legacy)

### Core Components
- **Image Encoder**: `ImageEncoderViT_hierarchical_moe` with MoE starting from layer 24
- **Mask Decoder**: `MaskDecoder_moe` with TwoWayTransformer using MoE
- **Training**: Distributed training with mixed precision (FP16/BF16)

## Common Development Commands

### Training
```bash
# Distributed training (multi-GPU)
torchrun --nproc_per_node=N train1.py --config configs/XinTong/10b_fsdp.yaml

# Single GPU training
python train1.py --config configs/XinTong/10b_fsdp.yaml
```

### Evaluation
```bash
# Evaluate IoU metrics
python eval_iou.py --config configs/XinTong/10b_fsdp.yaml

# Inference on large images
python inf_big_img.py --config configs/XinTong/10b_fsdp.yaml
```

### Testing
```bash
# Run test script
python test.py --config configs/XinTong/10b_fsdp.yaml
```

## Configuration System

### Config Files Location
- Main configs: `configs/XinTong/`
- Key configs:
  - `10b_fsdp.yaml`: 10B model with FSDP optimization
  - `10b_new.yaml`: Standard 10B model configuration
  - `1dot5B.yaml`: 1.5B baseline model

### Key Config Parameters
- `model.name`: Model architecture selector
- `model.args.encoder_mode`: Encoder configuration including MoE parameters
- `sam_checkpoint`: Path to pretrained SAM weights
- `train_dataset`/`val_dataset`: Dataset configuration
- `fsdp_settings`: FSDP configuration for memory optimization

## Dataset Structure

### Expected Data Format
- Images: RGB images (typically 512x512)
- Labels: Pixel-wise segmentation masks with class indices
- Classes: 33 land cover categories (see config files for complete list)

### Data Paths
- Training: Configured in `train_dataset.dataset.args.root_path_1/2`
- Validation: Configured in `val_dataset.dataset.args.root_path_1/2`

## Memory Optimization Features

### FSDP (Fully Sharded Data Parallel)
- Automatic parameter sharding across GPUs
- CPU offloading for inactive parameters
- Configured in `fsdp_settings` section

### Mixed Precision Training
- Uses `torch.cuda.amp.autocast()` for automatic mixed precision
- 8-bit optimizer (AdamW8bit) for memory efficiency
- Gradient scaling for numerical stability

### MoE Optimization
- Sparse activation with top-k expert selection
- Hierarchical routing to reduce computational overhead
- Expert weight initialization from 1.5B pretrained model

## Model Checkpoints

### Checkpoint Structure
- Saved in `save/` directory
- Format: `model_epoch_{epoch}.pth`
- Special checkpoints: `model_epoch_best.pth`, `model_epoch_last.pth`

### Loading Checkpoints
- Resume training: Set `resume: {epoch_number}` in config
- Pretrained weights: Configure `sam_checkpoint` path
- Supports partial loading with shape filtering

## Development Notes

### Training Process
1. **Weight Initialization**: First expert group uses exact 1.5B weights (frozen)
2. **Staged Training**: New experts initialized with noise + interpolation
3. **Memory Management**: Automatic GPU memory monitoring and cleanup
4. **Evaluation**: Comprehensive metrics including IoU, precision, recall, F1

### Key Model Files
- `models/sam.py`: Main model definitions
- `models/mmseg/models/sam/`: SAM component implementations
- `train1.py`: Training script with distributed support
- `eval_iou.py`: Evaluation metrics

### Performance Monitoring
- TensorBoard logging automatically enabled
- GPU memory usage tracking
- Comprehensive evaluation metrics table output

## Hardware Requirements

### Minimum Requirements
- GPU: Multiple GPUs recommended (8+ GB VRAM per GPU)
- RAM: 32GB+ system RAM
- Storage: 100GB+ for datasets and checkpoints

### Recommended Setup
- Multi-GPU setup with NVLink
- FSDP for memory optimization
- Mixed precision training enabled