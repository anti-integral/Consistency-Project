"""Tests for consistency model components."""

import pytest
import torch
import torch.nn as nn

from src.models.consistency_model import ConsistencyModel
from src.models.unet import UNetModel
from src.models.neural_operators import FourierConvBlock
from src.training.losses import PseudoHuberLoss


class TestConsistencyModel:
    """Test consistency model functionality."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple consistency model for testing."""
        # Create minimal UNet backbone
        backbone = UNetModel(
            in_channels=3,
            out_channels=3,
            channels=32,
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
            time_embed_dim=128,
        )

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

        return model

    def test_model_creation(self, simple_model):
        """Test model can be created."""
        assert isinstance(simple_model, ConsistencyModel)
        assert hasattr(simple_model, 'backbone')
        assert hasattr(simple_model, 'forward')
        assert hasattr(simple_model, 'sample')

    def test_forward_pass(self, simple_model):
        """Test forward pass works correctly."""
        batch_size = 4
        channels = 3
        size = 32

        # Create random input
        x = torch.randn(batch_size, channels, size, size)
        t = torch.rand(batch_size) * 1.5  # Random times

        # Forward pass
        output = simple_model(x, t)

        # Check output shape
        assert output.shape == (batch_size, channels, size, size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_sampling(self, simple_model):
        """Test sampling functionality."""
        batch_size = 2
        shape = (batch_size, 3, 32, 32)

        # Single-step sampling
        samples = simple_model.sample(
            shape=shape,
            num_steps=1,
            device=torch.device('cpu'),
        )

        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()

    def test_time_conversion(self, simple_model):
        """Test time-sigma conversions."""
        # Test sigma to t conversion
        sigma = torch.tensor([0.1, 1.0, 10.0])
        t = simple_model._sigma_to_t(sigma)

        # Convert back
        sigma_reconstructed = simple_model._t_to_sigma(t)

        # Check reconstruction
        assert torch.allclose(sigma, sigma_reconstructed, rtol=1e-5)

    def test_scalings(self, simple_model):
        """Test scaling computations."""
        sigma = torch.tensor([0.1, 1.0, 10.0])
        scalings = simple_model.get_scalings(sigma)

        # Check all scalings are computed
        assert 'c_skip' in scalings
        assert 'c_out' in scalings
        assert 'c_in' in scalings

        # Check shapes
        for key, value in scalings.items():
            assert value.shape == sigma.shape
            assert not torch.isnan(value).any()


class TestNeuralOperators:
    """Test neural operator components."""

    def test_fourier_conv_block(self):
        """Test Fourier convolution block."""
        in_channels = 64
        out_channels = 64
        modes = 16

        # Create block
        block = FourierConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            activation='gelu',
            norm=True,
            dropout=0.1,
        )

        # Test forward pass
        x = torch.randn(4, in_channels, 32, 32)
        output = block(x)

        assert output.shape == (4, out_channels, 32, 32)
        assert not torch.isnan(output).any()

    def test_spectral_conv(self):
        """Test spectral convolution operation."""
        from src.models.neural_operators import SpectralConv2d

        conv = SpectralConv2d(
            in_channels=32,
            out_channels=32,
            modes=8,
        )

        # Test with different input sizes
        for size in [16, 32, 64]:
            x = torch.randn(2, 32, size, size)
            output = conv(x)

            assert output.shape == x.shape
            assert not torch.isnan(output).any()


class TestLosses:
    """Test loss functions."""

    def test_pseudo_huber_loss(self):
        """Test Pseudo-Huber loss computation."""
        loss_fn = PseudoHuberLoss(c=0.01)

        # Test with random inputs
        input = torch.randn(4, 3, 32, 32)
        target = torch.randn(4, 3, 32, 32)

        loss = loss_fn(input, target)

        assert loss.ndim == 0  # Scalar loss
        assert loss >= 0  # Non-negative
        assert not torch.isnan(loss)

        # Test with identical inputs
        loss_same = loss_fn(input, input)
        assert torch.allclose(loss_same, torch.tensor(0.0), atol=1e-6)

    def test_loss_gradients(self):
        """Test loss gradients are well-behaved."""
        loss_fn = PseudoHuberLoss(c=0.01)

        # Create inputs requiring gradients
        input = torch.randn(2, 3, 16, 16, requires_grad=True)
        target = torch.randn(2, 3, 16, 16)

        # Compute loss and gradients
        loss = loss_fn(input, target)
        loss.backward()

        # Check gradients exist and are finite
        assert input.grad is not None
        assert not torch.isnan(input.grad).any()
        assert not torch.isinf(input.grad).any()


class TestTimeConditioning:
    """Test time conditioning modules."""

    def test_positional_embedding(self):
        """Test positional embeddings."""
        from src.models.time_conditioning import PositionalEmbedding

        embed_dim = 128
        pe = PositionalEmbedding(embed_dim)

        # Test with different time values
        t = torch.linspace(0, 1, 10)
        embeddings = pe(t)

        assert embeddings.shape == (10, embed_dim)
        assert not torch.isnan(embeddings).any()

        # Test that different times give different embeddings
        assert not torch.allclose(embeddings[0], embeddings[-1])

    def test_adaptive_group_norm(self):
        """Test adaptive group normalization."""
        from src.models.time_conditioning import AdaptiveGroupNorm

        norm = AdaptiveGroupNorm(
            num_groups=8,
            num_channels=64,
            time_embed_dim=128,
            use_scale_shift=True,
            use_pixel_norm=True,
        )

        # Test forward pass
        x = torch.randn(4, 64, 32, 32)
        time_emb = torch.randn(4, 128)

        output = norm(x, time_emb)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("num_steps", [1, 2, 5])
def test_consistency_property(batch_size, num_steps):
    """Test that consistency property holds."""
    # Create simple model
    backbone = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, 3, padding=1),
    )

    # Mock time embedding in backbone
    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = backbone

        def forward(self, x, t):
            # Simple time conditioning
            return self.conv(x) * (1 + t.view(-1, 1, 1, 1) * 0.1)

    model = ConsistencyModel(
        backbone=SimpleBackbone(),
        sigma_data=0.5,
        parameterization="v-prediction",
    )

    # Test sampling consistency
    shape = (batch_size, 3, 16, 16)
    samples1 = model.sample(shape, num_steps=num_steps)
    samples2 = model.sample(shape, num_steps=num_steps)

    # Samples should be different (due to different random noise)
    assert not torch.allclose(samples1, samples2)

    # But both should be valid images
    assert samples1.shape == shape
    assert samples2.shape == shape
    assert not torch.isnan(samples1).any()
    assert not torch.isnan(samples2).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])