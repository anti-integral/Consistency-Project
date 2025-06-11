#!/usr/bin/env python3
"""Script to generate samples from trained consistency model."""

import os
import sys
import argparse
from typing import Optional

import torch
import numpy as np
from PIL import Image
import torchvision
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import set_seed
from scripts.evaluate import create_model_from_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate samples from consistency model")

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
        "--output_dir",
        type=str,
        default="./generated_samples",
        help="Directory to save samples"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=64,
        help="Number of samples to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for generation"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=1,
        help="Number of sampling steps"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        help="Override image size"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--class_id",
        type=int,
        help="Class ID for conditional generation"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual images"
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        default=True,
        help="Save as grid"
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
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format"
    )

    return parser.parse_args()


def generate_samples(
    model,
    num_samples: int,
    batch_size: int,
    image_size: int,
    num_steps: int = 1,
    guidance_scale: float = 0.0,
    class_id: Optional[int] = None,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Generate samples from model.

    Args:
        model: Consistency model
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        image_size: Size of images
        num_steps: Number of sampling steps
        guidance_scale: CFG scale
        class_id: Optional class ID
        device: Device to use

    Returns:
        Generated samples tensor
    """
    samples = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating samples"):
            # Adjust last batch size
            current_batch_size = min(
                batch_size,
                num_samples - i * batch_size
            )

            # Generate class labels if needed
            if class_id is not None:
                class_labels = torch.full(
                    (current_batch_size,),
                    class_id,
                    dtype=torch.long,
                    device=device
                )
            else:
                class_labels = None

            # Generate samples
            batch_samples = model.sample(
                shape=(current_batch_size, 3, image_size, image_size),
                num_steps=num_steps,
                device=device,
                guidance_scale=guidance_scale,
                class_labels=class_labels,
            )

            samples.append(batch_samples.cpu())

    # Concatenate all samples
    samples = torch.cat(samples, dim=0)[:num_samples]

    return samples


def save_samples(
    samples: torch.Tensor,
    output_dir: str,
    save_individual: bool = False,
    save_grid: bool = True,
    format: str = "png",
):
    """Save generated samples.

    Args:
        samples: Generated samples tensor
        output_dir: Directory to save to
        save_individual: Whether to save individual images
        save_grid: Whether to save as grid
        format: Image format
    """
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize samples from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    if save_individual:
        # Save individual images
        individual_dir = os.path.join(output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)

        for i, sample in enumerate(samples):
            # Convert to PIL image
            img_array = (sample.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            # Save
            img_path = os.path.join(individual_dir, f"sample_{i:04d}.{format}")
            img.save(img_path)

        print(f"Saved {len(samples)} individual images to {individual_dir}")

    if save_grid:
        # Create grid
        nrow = int(np.sqrt(len(samples)))
        grid = torchvision.utils.make_grid(
            samples,
            nrow=nrow,
            padding=2,
            pad_value=1,
        )

        # Save grid
        grid_path = os.path.join(output_dir, f"sample_grid.{format}")
        torchvision.utils.save_image(grid, grid_path)
        print(f"Saved sample grid to {grid_path}")

    # Also save as tensor for further processing
    tensor_path = os.path.join(output_dir, "samples.pt")
    torch.save(samples, tensor_path)
    print(f"Saved raw tensor to {tensor_path}")


def create_interpolation(
    model,
    num_frames: int = 100,
    batch_size: int = 8,
    image_size: int = 32,
    num_steps: int = 1,
    device: torch.device = torch.device("cuda"),
    output_path: str = "interpolation.gif",
):
    """Create interpolation between random samples.

    Args:
        model: Consistency model
        num_frames: Number of interpolation frames
        batch_size: Number of interpolation paths
        image_size: Size of images
        num_steps: Sampling steps
        device: Device to use
        output_path: Path to save GIF
    """
    from PIL import Image

    # Generate start and end noise
    z_start = torch.randn(batch_size, 3, image_size, image_size, device=device)
    z_end = torch.randn(batch_size, 3, image_size, image_size, device=device)

    # Interpolate in noise space
    frames = []

    with torch.no_grad():
        for i in tqdm(range(num_frames), desc="Creating interpolation"):
            # Linear interpolation
            alpha = i / (num_frames - 1)
            z_interp = (1 - alpha) * z_start + alpha * z_end

            # Scale to max noise level
            z_interp = z_interp * model.sigma_max

            # Denoise
            t = torch.full(
                (batch_size,),
                model.t_max,
                device=device
            )
            samples = model.forward(z_interp, t)

            # Convert to image
            samples = (samples + 1) / 2
            samples = torch.clamp(samples, 0, 1)

            # Create grid
            grid = torchvision.utils.make_grid(
                samples,
                nrow=int(np.sqrt(batch_size)),
                padding=2,
                pad_value=1,
            )

            # Convert to PIL
            grid_np = (grid.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(Image.fromarray(grid_np))

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0
    )
    print(f"Saved interpolation to {output_path}")


def main():
    """Main generation function."""
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = create_model_from_checkpoint(
        args.checkpoint,
        args.config,
        device,
    )

    # Get image size
    if args.image_size is not None:
        image_size = args.image_size
    else:
        image_size = config['data']['image_size']

    print(f"Generating {args.num_samples} samples at {image_size}x{image_size}")
    print(f"Using {args.num_steps} sampling steps")

    # Generate samples
    samples = generate_samples(
        model,
        args.num_samples,
        args.batch_size,
        image_size,
        args.num_steps,
        args.guidance_scale,
        args.class_id,
        device,
    )

    # Save samples
    save_samples(
        samples,
        args.output_dir,
        args.save_individual,
        args.save_grid,
        args.format,
    )

    # Create interpolation if requested
    if args.num_samples >= 4:
        print("\nCreating interpolation...")
        interpolation_path = os.path.join(args.output_dir, "interpolation.gif")
        create_interpolation(
            model,
            num_frames=50,
            batch_size=min(9, args.num_samples),
            image_size=image_size,
            num_steps=args.num_steps,
            device=device,
            output_path=interpolation_path,
        )

    print("\nGeneration completed!")


if __name__ == "__main__":
    main()