#!/usr/bin/env python3
"""Main training script for Neural Operator Continuous Time Consistency Models."""

import os
import sys
import argparse
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import OmegaConf
import wandb

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.consistency_model import ConsistencyModel
from src.models.unet import UNetModel
from src.training.trainer import ConsistencyTrainer
from src.data.datasets import get_dataloader
from src.utils.helpers import (
    set_seed, setup_logger, create_exp_dir, save_config,
    count_parameters, get_model_size
)

# Optional wandb import
try:
    import wandb
except ImportError:
    wandb = None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Neural Operator Continuous Time Consistency Model"
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )

    # Overrides
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override number of iterations"
    )

    # Teacher model for distillation
    parser.add_argument(
        "--teacher_checkpoint",
        type=str,
        help="Path to teacher model checkpoint"
    )

    # Experiment
    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from"
    )

    # Resources
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        help="Comma-separated GPU IDs to use"
    )

    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def create_model(config: dict) -> ConsistencyModel:
    """Create model from config.

    Args:
        config: Model configuration

    Returns:
        Consistency model instance
    """
    # Create backbone
    backbone_config = config['model']['backbone']

    if backbone_config['type'] == 'unet':
        # Check if we should use simple UNet
        if backbone_config.get('use_simple', False):
            from src.models.simple_unet import SimpleUNetModel
            backbone = SimpleUNetModel(
                in_channels=backbone_config['in_channels'],
                out_channels=backbone_config['out_channels'],
                channels=backbone_config['channels'],
                channel_mult=backbone_config['channel_mult'],
                num_res_blocks=backbone_config['num_res_blocks'],
                time_embed_dim=config['model']['time_conditioning']['embed_dim'],
            )
        else:
            backbone = UNetModel(
                in_channels=backbone_config['in_channels'],
                out_channels=backbone_config['out_channels'],
                channels=backbone_config['channels'],
                channel_mult=backbone_config['channel_mult'],
                num_res_blocks=backbone_config['num_res_blocks'],
                attention_resolutions=backbone_config['attention_resolutions'],
                dropout=backbone_config['dropout'],
                use_checkpoint=backbone_config['use_checkpoint'],
                num_heads=backbone_config['num_heads'],
                num_head_channels=backbone_config['num_head_channels'],
                use_scale_shift_norm=backbone_config['use_scale_shift_norm'],
                resblock_updown=backbone_config['resblock_updown'],
                use_fno=config['model']['neural_operator']['use_fno'],
                fno_modes=config['model']['neural_operator'].get('fno_modes', 16),
                time_embed_dim=config['model']['time_conditioning']['embed_dim'],
            )
    else:
        raise ValueError(f"Unknown backbone type: {backbone_config['type']}")

    # Create consistency model
    consistency_config = config['model']['consistency']
    model = ConsistencyModel(
        backbone=backbone,
        sigma_data=consistency_config['sigma_data'],
        sigma_min=consistency_config['sigma_min'],
        sigma_max=consistency_config['sigma_max'],
        rho=consistency_config['rho'],
        parameterization=consistency_config['parameterization'],
        use_trigflow=config['model']['time_conditioning']['type'] == 'trigflow',
    )

    return model


def load_teacher_model(
    checkpoint_path: str,
    device: torch.device,
) -> nn.Module:
    """Load teacher diffusion model.

    Args:
        checkpoint_path: Path to teacher checkpoint
        device: Device to load to

    Returns:
        Teacher model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model (assuming same architecture)
    config = checkpoint['config']
    teacher = create_model(config)

    # Load weights
    teacher.load_state_dict(checkpoint['model_state_dict'])
    teacher.eval()

    return teacher


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Override config with command line args
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.optimizer.lr = args.lr
    if args.iterations is not None:
        config.training.total_iterations = args.iterations
    if args.teacher_checkpoint is not None:
        config.training.distillation.teacher_checkpoint = args.teacher_checkpoint

    # Set seed
    set_seed(args.seed)

    # Setup experiment directory
    exp_dir = create_exp_dir(
        config.paths.checkpoint_dir,
        args.exp_name
    )

    # Setup logging
    logger = setup_logger(
        'training',
        logging.DEBUG if args.debug else logging.INFO,
        os.path.join(exp_dir, 'train.log')
    )

    # Save config
    save_config(config, os.path.join(exp_dir, 'config.yaml'))

    # Setup device
    if args.gpu_ids:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        device = torch.device(f'cuda:{gpu_ids[0]}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"Using device: {device}")

    # Create model
    model = create_model(config)
    logger.info(f"Created model with {count_parameters(model):,} parameters")
    logger.info(f"Model size: {get_model_size(model):.2f} MB")

    # Load teacher if distillation mode
    teacher_model = None
    if config.training.mode == 'distillation':
        teacher_path = config.training.distillation.teacher_checkpoint
        if teacher_path is None:
            raise ValueError("Teacher checkpoint required for distillation")
        teacher_model = load_teacher_model(teacher_path, device)
        logger.info(f"Loaded teacher model from {teacher_path}")

    # Create data loaders
    train_loader = get_dataloader(config, split='train')
    val_loader = get_dataloader(config, split='val')
    logger.info(f"Created data loaders")

    # Create trainer
    trainer = ConsistencyTrainer(model, config, device)

    # Resume if specified
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.current_iteration = checkpoint['iteration']
        logger.info(f"Resumed from iteration {trainer.current_iteration}")

    # Train
    logger.info("Starting training...")
    try:
        trainer.train(train_loader, val_loader, teacher_model)
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise

    logger.info("Training completed!")

    # Final evaluation
    if val_loader is not None:
        from src.utils.metrics import compute_fid, compute_inception_score

        logger.info("Running final evaluation...")

        # FID score
        fid = compute_fid(
            model,
            val_loader,
            num_samples=10000,
            device=device,
            num_steps=config.sampling.num_steps,
        )
        logger.info(f"Final FID: {fid:.2f}")

        # Inception score
        is_mean, is_std = compute_inception_score(
            model,
            num_samples=10000,
            device=device,
            num_steps=config.sampling.num_steps,
        )
        logger.info(f"Final IS: {is_mean:.2f} ± {is_std:.2f}")

        # Log to wandb
        if config.wandb.enabled and wandb is not None:
            wandb.log({
                'final_fid': fid,
                'final_is_mean': is_mean,
                'final_is_std': is_std,
            })
        elif config.wandb.enabled:
            logger.warning("wandb requested but not installed")


if __name__ == "__main__":
    main()