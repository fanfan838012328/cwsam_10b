# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains CWSAM (Contrastive Vision-language Segment Anything Model), a vision segmentation model that extends the Segment Anything Model (SAM) architecture with advanced features including:

- **Mixture of Experts (MoE)** integration for improved parameter efficiency
- **Frequency-domain feature extraction** using FFT-based prompt generation
- **Adapter-based fine-tuning** for domain-specific tasks
- **Multi-scale segmentation** with 33 semantic classes for remote sensing applications

## Key Architecture Components

### Core Model Structure
- **Image Encoder**: Vision Transformer (ViT) with MoE layers starting from layer 28
- **Mask Decoder**: Enhanced decoder with skip connections and multi-class output
- **Prompt Encoder**: Frequency-domain prompt generation using FFT
- **SAM Integration**: Based on Meta's Segment Anything Model with custom extensions

### Model Variants
- `sam_moe_3b`: 3B parameter model with MoE layers (main production model)
- Standard SAM variants with different configurations

## Common Development Commands

### Training
```bash
# Distributed training (recommended)
python -m torch.distributed.launch --nproc_per_node=4 train1.py \
    --config configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml

# Single GPU training
python train1.py --config configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml
```

### Testing and Inference
```bash
# Model testing
python test.py --config configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml \
    --model path/to/model.pth

# Large image inference
python inf_big_img.py --config configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml \
    --model path/to/model.pth

# Evaluation with IoU metrics
python eval_iou.py --config configs/XinTong/XinTong_sam_vit_h_moe_3b.yaml \
    --model path/to/model.pth
```

## File Structure and Key Locations

### Model Implementation
- `models/sam.py`: Main SAM model registration and factory
- `models/mmseg/models/sam/`: Core SAM implementation
  - `image_encoder.py`: Standard ViT image encoder with adapters
  - `image_encoder_moe_layer.py`: MoE-enhanced ViT encoder
  - `mask_decoder.py`: Standard mask decoder
  - `mask_decoder_moe.py`: MoE-enhanced mask decoder
  - `common.py`: Shared utilities and building blocks

### Training and Data
- `train1.py`: Main training script with distributed training support
- `test.py`: Model testing and evaluation
- `datasets/`: Data loading and preprocessing
  - `datasets.py`: Dataset factory and utilities
  - `image_folder.py`: Image folder dataset implementation
- `utils.py`: Training utilities and helper functions

### Configuration
- `configs/XinTong/`: Model and training configurations
  - `XinTong_sam_vit_h_moe_3b.yaml`: Main 3B MoE model configuration
- Configuration defines:
  - 33 semantic classes for remote sensing
  - Data paths and preprocessing
  - Model architecture parameters
  - Training hyperparameters

## Model Architecture Details

### Image Encoder (ViT + MoE)
- **Base Architecture**: Vision Transformer with 32 layers
- **MoE Integration**: Starts from layer 28 (configurable via `moe_start_layer_index`)
- **MoE Parameters**: 
  - `moe_num_experts`: 128 experts per layer
  - `moe_k`: 4 experts selected per token
  - `moe_noisy_gating`: True for better expert selection
- **Adapter System**: FFT-based prompt generation with frequency domain features

### Mask Decoder
- **Multi-class Output**: Supports 33 semantic classes
- **Skip Connections**: Enhanced feature fusion
- **Upsampling**: ConvTranspose2d layers for mask resolution recovery

### Key Model Parameters
- Input size: 1024x1024 pixels
- Patch size: 16x16
- Embedding dimension: 1280 (ViT-Huge)
- Number of heads: 16
- MLP ratio: 4.0

## Training Configuration

### Dataset Structure
- **Classes**: 33 semantic classes including background, agricultural land, buildings, water bodies, etc.
- **Data Format**: RGB images with corresponding segmentation masks
- **Augmentation**: Configurable via training wrapper

### Training Parameters
- **Optimizer**: AdamW with learning rate 0.0002
- **Scheduler**: CosineAnnealingLR with minimum lr 1e-7
- **Batch Size**: 2 for training, 1 for validation
- **Epochs**: 200 maximum
- **Loss Function**: CrossEntropyLoss with IOU loss

### Distributed Training
- Uses `torch.distributed` for multi-GPU training
- DDP (DistributedDataParallel) for model parallelism
- Synchronized batch normalization across GPUs

## Development Best Practices

### Model Modifications
- When adding new model variants, register them in `models/sam.py`
- Use the `@register('model_name')` decorator for model factory registration
- Follow the existing naming convention: `sam_*` for SAM variants

### Training Customization
- Modify training parameters in YAML configuration files
- For new datasets, extend the dataset classes in `datasets/`
- Custom loss functions should be added to `models/mmseg/models/losses/`

### Memory Management
- Model uses gradient checkpointing for large models
- FFT operations are optimized for numerical stability
- MoE layers help reduce memory footprint compared to dense alternatives

### Debugging and Validation
- Use `print_model_parameters()` function to inspect trainable parameters
- Monitor training with distributed-aware logging
- Validation runs on single GPU (rank 0) to avoid redundant computation

## Common Issues and Solutions

### Training Issues
- **OOM (Out of Memory)**: Reduce batch size or use gradient accumulation
- **NaN in FFT**: The FFT implementation includes stability checks and fallbacks
- **Distributed Training**: Ensure proper NCCL backend initialization

### Model Loading
- Use `strict=False` when loading pretrained weights due to architecture differences
- Filter incompatible layers during weight loading
- SAM checkpoint loading includes shape matching validation

### Performance Optimization
- MoE layers provide better parameter efficiency than dense alternatives
- FFT-based prompt generation is computationally efficient
- Use mixed precision training for better performance

## Important Notes

- The model is designed for remote sensing segmentation tasks
- MoE integration significantly improves parameter efficiency
- FFT-based prompt generation provides frequency domain insights
- The codebase supports both standard and MoE variants of SAM
- All training scripts include proper distributed training support