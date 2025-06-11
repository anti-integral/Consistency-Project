"""Helper utilities for training and evaluation."""

import os
import json
import random
from typing import Dict, Optional, Any, Union
import logging

import torch
import torch.nn as nn
import numpy as np
from omegaconf import OmegaConf, DictConfig
import wandb


def set_seed(seed: int = 42):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(
    name: str = __name__,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Setup logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_checkpoint(
    checkpoint: Dict[str, Any],
    filepath: str,
    is_best: bool = False,
    best_filepath: Optional[str] = None,
):
    """Save model checkpoint.

    Args:
        checkpoint: Checkpoint dictionary
        filepath: Path to save checkpoint
        is_best: Whether this is the best model
        best_filepath: Path to save best model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)

    if is_best and best_filepath is not None:
        torch.save(checkpoint, best_filepath)


def load_checkpoint(
    filepath: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device('cuda'),
) -> Dict[str, Any]:
    """Load model checkpoint.

    Args:
        filepath: Path to checkpoint
        model: Optional model to load state into
        optimizer: Optional optimizer to load state into
        device: Device to load to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)

    if model is not None and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint


def count_parameters(model: nn.Module) -> int:
    """Count number of trainable parameters.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> float:
    """Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Model size in MB
    """
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_mb = (param_size + buffer_size) / 1024 / 1024

    return size_mb


def create_exp_dir(
    base_dir: str,
    exp_name: Optional[str] = None,
) -> str:
    """Create experiment directory.

    Args:
        base_dir: Base directory for experiments
        exp_name: Optional experiment name

    Returns:
        Path to experiment directory
    """
    if exp_name is None:
        from datetime import datetime
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create subdirectories
    for subdir in ['checkpoints', 'samples', 'logs', 'configs']:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)

    return exp_dir


def save_config(
    config: Union[Dict, DictConfig],
    filepath: str,
):
    """Save configuration to file.

    Args:
        config: Configuration dictionary or OmegaConf config
        filepath: Path to save config
    """
    if isinstance(config, DictConfig):
        OmegaConf.save(config, filepath)
    else:
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)


def load_config(filepath: str) -> DictConfig:
    """Load configuration from file.

    Args:
        filepath: Path to config file

    Returns:
        Configuration object
    """
    if filepath.endswith('.yaml') or filepath.endswith('.yml'):
        return OmegaConf.load(filepath)
    else:
        with open(filepath, 'r') as f:
            return OmegaConf.create(json.load(f))


def get_device(gpu_id: Optional[int] = None) -> torch.device:
    """Get compute device.

    Args:
        gpu_id: Optional specific GPU ID

    Returns:
        Device object
    """
    if torch.cuda.is_available():
        if gpu_id is not None:
            return torch.device(f'cuda:{gpu_id}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self, name: str = 'meter', fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    """Display progress of multiple meters."""

    def __init__(self, num_batches: int, meters: list, prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def grad_norm(model: nn.Module) -> float:
    """Compute gradient norm.

    Args:
        model: Model to compute gradient norm for

    Returns:
        Gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def log_wandb_images(
    images: torch.Tensor,
    caption: str = "samples",
    num_images: int = 16,
    normalize: bool = True,
):
    """Log images to wandb.

    Args:
        images: Tensor of images
        caption: Caption for images
        num_images: Number of images to log
        normalize: Whether to normalize images
    """
    if not wandb.run:
        return

    # Select subset
    images = images[:num_images]

    # Convert to grid
    grid = torchvision.utils.make_grid(
        images,
        nrow=int(np.sqrt(num_images)),
        normalize=normalize,
        value_range=(-1, 1) if normalize else (0, 1),
    )

    # Log
    wandb.log({caption: wandb.Image(grid)})


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = 'nccl',
):
    """Setup distributed training.

    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Distribution backend
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize process group
    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # Set device
    torch.cuda.set_device(rank)


def cleanup_distributed():
    """Cleanup distributed training."""
    torch.distributed.destroy_process_group()