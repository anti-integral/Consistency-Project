# Project Structure

## Overview
Neural Operator Continuous Time Consistency Model (NO-CTCM) combines Fourier Neural Operators with continuous-time consistency models for fast, high-quality image generation.

## Directory Structure

```
neural-consistency-model/
├── configs/                      # Configuration files
│   ├── cifar10_ctcm.yaml        # Main CIFAR-10 config
│   ├── cifar10_distillation.yaml # Distillation config
│   └── cifar10_training.yaml    # Training from scratch config
│
├── src/                         # Source code
│   ├── __init__.py
│   │
│   ├── models/                  # Model architectures
│   │   ├── __init__.py
│   │   ├── consistency_model.py # Main consistency model
│   │   ├── unet.py             # U-Net backbone
│   │   ├── neural_operators.py  # FNO layers
│   │   ├── time_conditioning.py # TrigFlow time embeddings
│   │   └── ema.py              # Exponential moving average
│   │
│   ├── training/                # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main trainer class
│   │   ├── losses.py           # Loss functions
│   │   ├── schedulers.py       # Training schedules
│   │   └── distillation.py     # Distillation utilities
│   │
│   ├── data/                    # Data loading
│   │   ├── __init__.py
│   │   └── datasets.py         # Dataset classes
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       ├── metrics.py          # Evaluation metrics
│       └── helpers.py          # Helper functions
│
├── scripts/                     # Executable scripts
│   ├── __init__.py
│   ├── train_consistency.py     # Main training script
│   ├── evaluate.py             # Evaluation script
│   └── generate_samples.py     # Sample generation
│
├── setup.py                     # Package setup
├── requirements.txt             # Dependencies
├── README.md                    # Main documentation
└── PROJECT_STRUCTURE.md         # This file
```

## Key Components

### Models
- **ConsistencyModel**: Main model class implementing continuous-time consistency
- **UNetModel**: U-Net backbone with FNO integration
- **FourierConvBlock**: Spectral convolution layers for trajectory learning
- **TrigFlowTimeConditioning**: Stable time embeddings from TrigFlow

### Training
- **ConsistencyTrainer**: Handles both distillation and from-scratch training
- **PseudoHuberLoss**: Robust loss function without LPIPS bias
- **ProgressiveSchedule**: Discretization doubling during training
- **ECTDistiller**: Efficient distillation from pre-trained models

### Features
- TrigFlow parameterization for stable training
- Fourier Neural Operators for efficient trajectory modeling
- Progressive training schedules
- Multi-scale consistency losses
- EMA with theoretical fixes

## Usage Examples

### Training from scratch
```bash
python scripts/train_consistency.py --config configs/cifar10_training.yaml
```

### Distillation from diffusion model
```bash
python scripts/train_consistency.py \
    --config configs/cifar10_distillation.yaml \
    --teacher_checkpoint path/to/diffusion_model.pt
```

### Evaluation
```bash
python scripts/evaluate.py \
    --checkpoint path/to/model.pt \
    --num_samples 50000 \
    --metrics fid is precision recall
```

### Generation
```bash
python scripts/generate_samples.py \
    --checkpoint path/to/model.pt \
    --num_samples 64 \
    --num_steps 1
```

## Configuration System

The project uses Hydra/OmegaConf for configuration management. Key configuration sections:

- `model`: Architecture settings (backbone, FNO, time conditioning)
- `training`: Training hyperparameters (optimizer, loss, schedules)
- `data`: Dataset configuration
- `sampling`: Generation settings
- `paths`: Directory paths

## Performance Optimizations

1. **Fourier Neural Operators**: Efficient learning in frequency space
2. **Progressive Training**: Start with few discretization steps, gradually increase
3. **Mixed Precision**: Optional FP16 training for memory efficiency
4. **EMA Optimization**: Improved update strategy from TrigFlow
5. **Batch Processing**: Efficient trajectory computation

## Extension Points

The codebase is designed to be easily extensible:

1. **New Architectures**: Inherit from `nn.Module` and use in `ConsistencyModel`
2. **Custom Losses**: Add to `src/training/losses.py`
3. **New Datasets**: Implement dataset class in `src/data/datasets.py`
4. **Alternative Schedules**: Add to `src/training/schedulers.py`

## Troubleshooting

Common issues and solutions:

1. **Training Instability**: Enable progressive training, reduce learning rate
2. **Out of Memory**: Reduce batch size, enable gradient checkpointing
3. **Poor Sample Quality**: Increase training iterations, tune loss weights
4. **Slow Training**: Enable mixed precision, use ECT distillation

## References

Key papers implemented:
- TrigFlow (OpenAI, 2024)
- Easy Consistency Tuning (CMU, 2024)
- Consistency Trajectory Models (Sony, 2023)
- Fourier Neural Operators (Caltech, 2021)