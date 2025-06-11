#!/usr/bin/env python3
"""Create a fixed cifar10_ctcm.yaml config."""

import yaml
from pathlib import Path

# Complete config with all required fields
config = {
    'model': {
        'type': 'NO-CTCM',
        'backbone': {
            'type': 'unet',
            'use_simple': False,
            'in_channels': 3,
            'out_channels': 3,
            'channels': 128,
            'channel_mult': [1, 2, 2, 2],
            'num_res_blocks': 2,
            'attention_resolutions': [16, 8],
            'dropout': 0.0,
            'use_checkpoint': False,
            'num_heads': 4,
            'num_head_channels': 64,
            'use_scale_shift_norm': True,
            'resblock_updown': True
        },
        'neural_operator': {
            'use_fno': True,
            'fno_modes': 16,
            'fno_width': 128,
            'fno_layers': 4,
            'spectral_normalization': True
        },
        'time_conditioning': {
            'type': 'trigflow',
            'embed_dim': 256,
            'positional_encoding': True,
            'fourier_scale': 1.0,
            'adaptive_norm': True,
            'pixel_norm': True
        },
        'consistency': {
            'parameterization': 'v-prediction',
            'sigma_data': 0.5,
            'sigma_min': 0.002,
            'sigma_max': 80.0,
            'rho': 7.0
        }
    },
    'data': {
        'dataset': 'cifar10',
        'image_size': 32,
        'channels': 3,
        'flip_prob': 0.5,
        'normalize': True,
        'mean': [0.5, 0.5, 0.5],
        'std': [0.5, 0.5, 0.5],
        'num_workers': 4
    },
    'training': {
        'mode': 'consistency_training',  # THIS IS THE KEY FIELD
        'batch_size': 128,
        'total_iterations': 100000,
        'optimizer': {
            'type': 'adamw',
            'lr': 0.0003,
            'betas': [0.9, 0.99],
            'weight_decay': 0.01,
            'eps': 1e-8
        },
        'lr_scheduler': {
            'type': 'cosine',
            'warmup_steps': 5000,
            'min_lr': 1e-6
        },
        'loss': {
            'type': 'pseudo_huber',
            'c': 0.01,
            'reduction': 'mean'
        },
        'progressive': {
            'enabled': True,
            'initial_discretization': 18,
            'final_discretization': 1280,
            'doubling_iterations': 20000
        },
        'ema': {
            'enabled': True,
            'decay': 0.9999,
            'update_every': 10
        },
        'gradient_clip': 1.0,
        'checkpoint': {
            'save_every': 5000,
            'keep_last': 5
        },
        'log_every': 100,
        'sample_every': 1000,
        'distributed': {
            'enabled': False,
            'backend': 'nccl'
        }
    },
    'sampling': {
        'num_steps': 1,
        'solver': 'euler',
        'guidance_scale': 0.0
    },
    'evaluation': {
        'batch_size': 256,
        'num_samples': 50000,
        'metrics': ['fid', 'is', 'precision', 'recall']
    },
    'wandb': {
        'enabled': False,
        'project': 'neural-consistency-model',
        'entity': None,
        'tags': ['cifar10', 'ctcm', 'fno']
    },
    'paths': {
        'data_dir': './data',
        'checkpoint_dir': './checkpoints',
        'sample_dir': './samples',
        'log_dir': './logs'
    }
}

# Save the config
config_path = Path('configs/cifar10_ctcm_fixed.yaml')
config_path.parent.mkdir(exist_ok=True)

with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print(f"Created fixed config at: {config_path}")
print("\nYou can now run:")
print(f"python scripts/train_consistency.py --config {config_path}")

# Also create a smaller version for testing
config_small = config.copy()
config_small['model']['backbone']['channels'] = 64
config_small['model']['backbone']['channel_mult'] = [1, 2, 2]
config_small['training']['batch_size'] = 32
config_small['training']['total_iterations'] = 10000

config_small_path = Path('configs/cifar10_ctcm_small.yaml')
with open(config_small_path, 'w') as f:
    yaml.dump(config_small, f, default_flow_style=False, sort_keys=False)

print(f"\nAlso created smaller config at: {config_small_path}")
print(f"python scripts/train_consistency.py --config {config_small_path}")