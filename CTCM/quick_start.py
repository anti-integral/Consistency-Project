#!/usr/bin/env python3
"""Quick start script to test the installation and run a minimal training."""

import os
import sys
import torch
import argparse
from pathlib import Path

def check_environment():
    """Check if environment is properly set up."""
    print("🔍 Checking environment...")

    # Check Python version
    print(f"✓ Python version: {sys.version}")

    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA version: {torch.version.cuda}")
            print(f"  - GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

    # Check NumPy version
    try:
        import numpy as np
        print(f"✓ NumPy version: {np.__version__}")
        if int(np.__version__.split('.')[0]) >= 2:
            print("⚠️  Warning: NumPy 2.x detected. PyTorch may have compatibility issues.")
            print("   Run: pip install 'numpy<2.0.0'")
    except ImportError:
        print("❌ NumPy not installed!")
        return False

    # Check other dependencies
    required_packages = ['torchvision', 'omegaconf', 'einops', 'tqdm']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed!")
            return False

    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    dirs = ['data', 'checkpoints', 'samples', 'logs', 'experiments']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")


def download_sample_data():
    """Download a small sample of CIFAR-10 for testing."""
    print("\n📥 Preparing CIFAR-10 dataset...")
    try:
        import torchvision
        # This will download CIFAR-10 if not already present
        dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True
        )
        print(f"✓ CIFAR-10 ready ({len(dataset)} training samples)")
        return True
    except Exception as e:
        print(f"❌ Failed to prepare CIFAR-10: {e}")
        return False


def run_minimal_test():
    """Run a minimal test to ensure model can be created."""
    print("\n🧪 Testing model creation...")
    try:
        # Add src to path
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        from src.models.consistency_model import ConsistencyModel
        from src.models.unet import UNetModel

        # Create minimal model
        backbone = UNetModel(
            in_channels=3,
            out_channels=3,
            channels=32,  # Small for testing
            channel_mult=[1, 2],
            num_res_blocks=1,
            attention_resolutions=[],
            dropout=0.0,
            use_checkpoint=False,
            num_heads=1,
            num_head_channels=32,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_fno=False,
            time_embed_dim=128,  # Ensure this matches
        )

        model = ConsistencyModel(
            backbone=backbone,
            sigma_data=0.5,
            sigma_min=0.002,
            sigma_max=80.0,
            rho=7.0,
            parameterization="v-prediction",
            use_trigflow=True,
        )

        # Test forward pass
        x = torch.randn(2, 3, 32, 32)
        t = torch.rand(2)
        output = model(x, t)

        print(f"✓ Model created successfully")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        return True

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_minimal_config():
    """Create a minimal training config for testing."""
    print("\n📝 Creating minimal test configuration...")

    minimal_config = """# Minimal configuration for testing
model:
  type: "NO-CTCM"
  backbone:
    type: "unet"
    in_channels: 3
    out_channels: 3
    channels: 32  # Small for testing
    channel_mult: [1, 2]
    num_res_blocks: 1
    attention_resolutions: []
    dropout: 0.0
    use_checkpoint: false
    num_heads: 1
    num_head_channels: 32
    use_scale_shift_norm: false
    resblock_updown: false

  neural_operator:
    use_fno: false

  time_conditioning:
    type: "trigflow"
    embed_dim: 128  # Match the model
    positional_encoding: true

  consistency:
    parameterization: "v-prediction"
    sigma_data: 0.5
    sigma_min: 0.002
    sigma_max: 80.0
    rho: 7.0

data:
  dataset: "cifar10"
  image_size: 32
  channels: 3
  flip_prob: 0.5
  normalize: true

training:
  mode: "consistency_training"
  batch_size: 32
  total_iterations: 100  # Very short for testing

  optimizer:
    type: "adamw"
    lr: 1e-4
    betas: [0.9, 0.99]
    weight_decay: 1e-2

  lr_scheduler:
    type: "cosine"
    warmup_steps: 10
    min_lr: 1e-6

  loss:
    type: "pseudo_huber"
    c: 0.01

  progressive:
    enabled: false  # Disable for quick test

  ema:
    enabled: true
    decay: 0.999

  gradient_clip: 1.0

  checkpoint:
    save_every: 50

  log_every: 10
  sample_every: 50

sampling:
  num_steps: 1

paths:
  data_dir: "./data"
  checkpoint_dir: "./checkpoints/test"
  sample_dir: "./samples/test"
  log_dir: "./logs/test"

wandb:
  enabled: false  # Disable for testing
"""

    config_path = Path("configs/test_minimal.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        f.write(minimal_config)

    print(f"✓ Created {config_path}")
    return str(config_path)


def main():
    """Run quick start checks and setup."""
    parser = argparse.ArgumentParser(description="Quick start for NO-CTCM")
    parser.add_argument("--run-test", action="store_true", help="Run minimal training test")
    args = parser.parse_args()

    print("🚀 Neural Operator Continuous Time Consistency Model - Quick Start")
    print("="*60)

    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please install missing packages.")
        print("   Run: pip install -r requirements.txt")
        return 1

    # Create directories
    create_directories()

    # Download data
    if not download_sample_data():
        return 1

    # Test model
    if not run_minimal_test():
        return 1

    print("\n✅ All checks passed!")

    if args.run_test:
        # Create minimal config
        config_path = create_minimal_config()

        print("\n🏃 Running minimal training test...")
        print(f"Command: python scripts/train_consistency.py --config {config_path}")

        # Run training
        import subprocess
        try:
            result = subprocess.run([
                sys.executable,
                "scripts/train_consistency.py",
                "--config", config_path,
                "--exp_name", "quick_test"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print("\n✅ Training test completed successfully!")
            else:
                print(f"\n❌ Training failed with error:")
                print(result.stderr)

        except Exception as e:
            print(f"\n❌ Failed to run training: {e}")

    else:
        print("\n📖 Next steps:")
        print("1. Run minimal training test:")
        print("   python quick_start.py --run-test")
        print("\n2. Train from scratch on CIFAR-10:")
        print("   python scripts/train_consistency.py --config configs/cifar10_training.yaml")
        print("\n3. Generate samples:")
        print("   python scripts/generate_samples.py --checkpoint path/to/model.pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())