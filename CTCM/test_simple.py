#!/usr/bin/env python3
"""Simple test script to verify basic functionality."""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.consistency_model import ConsistencyModel


class SimpleUNet(nn.Module):
    """Simplified UNet for testing."""

    def __init__(self, channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

        self.input_conv = nn.Conv2d(3, channels, 3, padding=1)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(8, channels),
                nn.SiLU(),
                nn.Conv2d(channels, channels, 3, padding=1),
            )
            for _ in range(4)
        ])

        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x, t):
        # Embed time
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        t_emb = self.time_embed(t)

        # Process image
        h = self.input_conv(x)

        for block in self.blocks:
            h = h + block(h)

        return self.output_conv(h)


def test_simple_model():
    """Test a simple consistency model."""
    print("Testing simple consistency model...")

    # Create simple backbone
    backbone = SimpleUNet(channels=64)

    # Create consistency model
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
    batch_size = 2
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.rand(batch_size)

    try:
        output = model(x, t)
        print(f"✓ Forward pass successful!")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test sampling
        print("\nTesting sampling...")
        samples = model.sample(
            shape=(2, 3, 32, 32),
            num_steps=1,
            device=torch.device('cpu'),
        )
        print(f"✓ Sampling successful!")
        print(f"  Samples shape: {samples.shape}")
        print(f"  Samples range: [{samples.min():.2f}, {samples.max():.2f}]")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_original_model():
    """Test the original model with correct dimensions."""
    print("\nTesting original UNet model...")

    from src.models.unet import UNetModel

    # Create model with consistent dimensions
    backbone = UNetModel(
        in_channels=3,
        out_channels=3,
        channels=64,  # Base channels
        channel_mult=[1, 2, 2],  # Results in 64, 128, 128 channels
        num_res_blocks=2,
        attention_resolutions=[16],
        dropout=0.0,
        use_checkpoint=False,
        num_heads=4,
        num_head_channels=32,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fno=False,
        time_embed_dim=256,  # Larger time embedding
    )

    # Test just the backbone first
    x = torch.randn(2, 3, 32, 32)
    t = torch.rand(2)

    try:
        # Get time embedding shape
        time_emb = backbone.time_embed(t)
        print(f"Time embedding shape: {time_emb.shape}")

        # Test forward
        output = backbone(x, t)
        print(f"✓ UNet forward pass successful!")
        print(f"  Output shape: {output.shape}")

        # Now test in consistency model
        model = ConsistencyModel(
            backbone=backbone,
            sigma_data=0.5,
            parameterization="v-prediction",
            use_trigflow=True,
        )

        output = model(x, t)
        print(f"✓ Consistency model forward pass successful!")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Consistency Model Simple Tests")
    print("=" * 60)

    # Test simple model first
    success1 = test_simple_model()

    # Test original model
    success2 = test_original_model()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ All tests passed!")
        print("\nYou can now run training with:")
        print("python scripts/train_consistency.py --config configs/cifar10_training.yaml")
    else:
        print("❌ Some tests failed. Please check the errors above.")

    print("=" * 60)