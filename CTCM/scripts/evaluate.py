#!/usr/bin/env python3
"""Evaluation script for consistency models."""

import os
import sys
import argparse
import logging
from typing import Dict, Optional

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.consistency_model import ConsistencyModel
from src.models.unet import UNetModel
from src.data.datasets import get_dataloader
from src.utils.metrics import MetricsCalculator, compute_fid, compute_inception_score
from src.utils.helpers import setup_logger, load_checkpoint, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate consistency model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config (if not in checkpoint)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to dataset"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=50000,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["fid", "is", "precision", "recall"],
        help="Metrics to compute"
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save generated samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    return parser.parse_args()


def create_model_from_checkpoint(
    checkpoint_path: str,
    config_path: Optional[str] = None,
    device: torch.device = torch.device("cuda"),
) -> ConsistencyModel:
    """Create model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        config_path: Optional path to config
        device: Device to load to

    Returns:
        Loaded model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config
    if config_path is not None:
        config = OmegaConf.load(config_path)
    elif 'config' in checkpoint:
        config = checkpoint['config']
    else:
        raise ValueError("Config not found in checkpoint and not provided")

    # Create model
    from scripts.train_consistency import create_model
    model = create_model(config)

    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'ema_state_dict' in checkpoint:
        # Load EMA weights if available
        from src.models.ema import ExponentialMovingAverage
        ema = ExponentialMovingAverage(model)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model)
    else:
        raise ValueError("No model weights found in checkpoint")

    model.to(device)
    model.eval()

    return model, config


def evaluate_model(
    model: ConsistencyModel,
    config: Dict,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Evaluate model on various metrics.

    Args:
        model: Model to evaluate
        config: Configuration
        args: Command line arguments

    Returns:
        Dictionary of metric results
    """
    device = torch.device(args.device)
    results = {}

    # Create metrics calculator
    calculator = MetricsCalculator(device)

    # Get reference data loader
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    reference_loader = get_dataloader(config, split='val', shuffle=False)

    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    generated_samples = []

    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size

    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch_size = min(args.batch_size, args.num_samples - i * args.batch_size)

            samples = model.sample(
                shape=(batch_size, 3, config['data']['image_size'], config['data']['image_size']),
                num_steps=args.num_steps,
                device=device,
            )

            generated_samples.append(samples.cpu())

    generated_samples = torch.cat(generated_samples, dim=0)[:args.num_samples]

    # Get reference samples
    print("Loading reference samples...")
    reference_samples = []

    for batch in reference_loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        reference_samples.append(images)

        if len(reference_samples) * images.shape[0] >= args.num_samples:
            break

    reference_samples = torch.cat(reference_samples, dim=0)[:args.num_samples]

    # Compute metrics
    if "fid" in args.metrics:
        print("Computing FID...")
        fid = calculator.calculate_fid(
            reference_samples,
            generated_samples,
            batch_size=args.batch_size,
        )
        results['fid'] = fid
        print(f"FID: {fid:.2f}")

    if "is" in args.metrics:
        print("Computing Inception Score...")
        is_mean, is_std = calculator.calculate_inception_score(
            generated_samples,
            batch_size=args.batch_size,
        )
        results['is_mean'] = is_mean
        results['is_std'] = is_std
        print(f"IS: {is_mean:.2f} ± {is_std:.2f}")

    if "precision" in args.metrics or "recall" in args.metrics:
        print("Computing Precision/Recall...")

        # Extract features
        print("Extracting features...")
        real_features = calculator._extract_features(
            reference_samples, args.batch_size
        )
        fake_features = calculator._extract_features(
            generated_samples, args.batch_size
        )

        precision, recall = calculator.calculate_precision_recall(
            real_features, fake_features
        )

        if "precision" in args.metrics:
            results['precision'] = precision
            print(f"Precision: {precision:.3f}")

        if "recall" in args.metrics:
            results['recall'] = recall
            print(f"Recall: {recall:.3f}")

    # Save samples if requested
    if args.save_samples:
        save_path = os.path.join(args.output_dir, "generated_samples.pt")
        torch.save(generated_samples, save_path)
        print(f"Saved samples to {save_path}")

        # Also save as images
        import torchvision
        grid = torchvision.utils.make_grid(
            generated_samples[:64],
            nrow=8,
            normalize=True,
            value_range=(-1, 1),
        )

        img_path = os.path.join(args.output_dir, "sample_grid.png")
        torchvision.utils.save_image(grid, img_path)
        print(f"Saved sample grid to {img_path}")

    return results


def plot_sampling_speed(
    model: ConsistencyModel,
    config: Dict,
    output_dir: str,
    max_steps: int = 10,
    device: torch.device = torch.device("cuda"),
):
    """Plot sample quality vs number of steps.

    Args:
        model: Model to evaluate
        config: Configuration
        output_dir: Directory to save plots
        max_steps: Maximum number of steps to test
        device: Device to use
    """
    print("Evaluating sampling speed vs quality...")

    steps_list = list(range(1, max_steps + 1))
    fid_scores = []

    # Get reference loader
    reference_loader = get_dataloader(config, split='val', shuffle=False)

    # Get reference samples
    reference_samples = []
    for batch in reference_loader:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        reference_samples.append(images)
        if len(reference_samples) * images.shape[0] >= 10000:
            break
    reference_samples = torch.cat(reference_samples, dim=0)[:10000]

    # Calculate FID for different step counts
    calculator = MetricsCalculator(device)

    for steps in tqdm(steps_list, desc="Testing step counts"):
        # Generate samples
        generated = []
        for _ in range(10000 // 100):
            samples = model.sample(
                shape=(100, 3, config['data']['image_size'], config['data']['image_size']),
                num_steps=steps,
                device=device,
            )
            generated.append(samples.cpu())

        generated = torch.cat(generated, dim=0)[:10000]

        # Calculate FID
        fid = calculator.calculate_fid(reference_samples, generated, batch_size=100)
        fid_scores.append(fid)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(steps_list, fid_scores, 'b-o', linewidth=2, markersize=8)
    plt.xlabel('Number of Sampling Steps', fontsize=12)
    plt.ylabel('FID Score', fontsize=12)
    plt.title('Sample Quality vs Sampling Steps', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(steps_list)

    # Highlight single-step performance
    plt.axhline(y=fid_scores[0], color='r', linestyle='--', alpha=0.5, label=f'Single-step FID: {fid_scores[0]:.2f}')
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'sampling_speed_vs_quality.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"Saved plot to {plot_path}")

    # Save raw data
    data = {
        'steps': steps_list,
        'fid_scores': fid_scores,
    }
    torch.save(data, os.path.join(output_dir, 'sampling_speed_data.pt'))


def main():
    """Main evaluation function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    logger = setup_logger(
        'evaluation',
        logging.INFO,
        os.path.join(args.output_dir, 'evaluate.log')
    )

    logger.info(f"Evaluation arguments: {args}")

    # Load model
    device = torch.device(args.device)
    logger.info(f"Loading model from {args.checkpoint}")

    model, config = create_model_from_checkpoint(
        args.checkpoint,
        args.config,
        device,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    results = evaluate_model(model, config, args)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Plot sampling speed if requested
    if args.num_steps == 1:  # Only plot if evaluating single-step model
        plot_sampling_speed(model, config, args.output_dir, device=device)

    logger.info("Evaluation completed!")

    # Print summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    for metric, value in results.items():
        if isinstance(value, float):
            print(f"{metric.upper()}: {value:.3f}")
        else:
            print(f"{metric.upper()}: {value}")
    print("="*50)


if __name__ == "__main__":
    main()