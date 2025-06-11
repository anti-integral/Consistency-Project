"""Fourier Neural Operator layers for trajectory modeling.

Implements spectral convolutions for efficient learning in frequency space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import numpy as np


class SpectralConv2d(nn.Module):
    """2D Spectral convolution layer."""

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes

        # Initialize weights in Fourier space
        self.scale = 1 / (in_channels * out_channels)

        # Complex weights for Fourier modes
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(in_channels, out_channels, modes, modes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spectral convolution.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        B, C, H, W = x.shape

        # Compute FFT
        x_ft = torch.fft.rfft2(x, norm='ortho')

        # Truncate to relevant modes
        x_ft = x_ft[..., :self.modes, :self.modes]

        # Multiply by weights in Fourier space
        weights = torch.complex(self.weights_real, self.weights_imag)

        # Einsum multiplication
        out_ft = torch.einsum('bcxy,cdxy->bdxy', x_ft, weights)

        # Pad back to original size
        out_ft_padded = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.complex64, device=x.device
        )
        out_ft_padded[..., :self.modes, :self.modes] = out_ft

        # Inverse FFT
        out = torch.fft.irfft2(out_ft_padded, s=(H, W), norm='ortho')

        return out


class FourierConvBlock(nn.Module):
    """Fourier Neural Operator block with residual connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        activation: str = 'gelu',
        norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # Spectral convolution
        self.spectral_conv = SpectralConv2d(in_channels, out_channels, modes)

        # Regular convolution branch (for high frequencies)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        # Normalization
        self.norm = nn.GroupNorm(8, out_channels) if norm else nn.Identity()

        # Activation
        activations = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
        }
        self.activation = activations.get(activation, nn.GELU())

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through FNO block.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Output tensor of shape [B, C', H, W]
        """
        # Spectral path (low frequencies)
        out_spectral = self.spectral_conv(x)

        # Conv path (high frequencies)
        out_conv = self.conv(x)

        # Combine paths
        out = out_spectral + out_conv

        # Apply norm, activation, dropout
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Residual connection
        out = out + self.skip(x)

        return out


class TemporalFourierOperator(nn.Module):
    """Temporal Fourier operator for trajectory modeling.

    Maps initial conditions to full trajectories in Fourier space.
    """

    def __init__(
        self,
        spatial_size: int,
        channels: int,
        time_steps: int,
        modes_spatial: int = 16,
        modes_temporal: int = 8,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.spatial_size = spatial_size
        self.channels = channels
        self.time_steps = time_steps
        self.modes_spatial = modes_spatial
        self.modes_temporal = modes_temporal

        # Lifting layer
        self.lift = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, 1),
            nn.GELU(),
        )

        # Fourier layers operating on space-time
        self.fourier_layers = nn.ModuleList([
            SpectralConv3d(hidden_dim, hidden_dim, modes_spatial, modes_temporal)
            for _ in range(4)
        ])

        # Projection layer
        self.project = nn.Conv2d(hidden_dim, channels, 1)

    def forward(self, x_0: torch.Tensor, time_points: torch.Tensor) -> torch.Tensor:
        """Map initial condition to trajectory at specified time points.

        Args:
            x_0: Initial condition, shape [B, C, H, W]
            time_points: Time points to evaluate, shape [T]

        Returns:
            Trajectory tensor of shape [B, T, C, H, W]
        """
        B, C, H, W = x_0.shape
        T = len(time_points)

        # Lift to hidden dimension
        h = self.lift(x_0)  # [B, hidden, H, W]

        # Expand temporally
        h = repeat(h, 'b c h w -> b c t h w', t=T)

        # Add time encoding
        time_enc = self._get_time_encoding(time_points, H, W).to(h.device)
        h = h + time_enc.unsqueeze(0)

        # Apply Fourier layers
        for layer in self.fourier_layers:
            h = layer(h) + h  # Residual connection

        # Project back to original channels
        h = rearrange(h, 'b c t h w -> (b t) c h w')
        out = self.project(h)
        out = rearrange(out, '(b t) c h w -> b t c h w', b=B, t=T)

        return out

    def _get_time_encoding(
        self,
        time_points: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """Generate sinusoidal time encodings."""
        # Create position encodings
        freqs = torch.linspace(0, 1, self.modes_temporal // 2)
        freqs = torch.pow(10000, -freqs)

        # Time encodings
        t_enc = time_points.unsqueeze(-1) @ freqs.unsqueeze(0)
        t_enc = torch.cat([torch.sin(t_enc), torch.cos(t_enc)], dim=-1)

        # Reshape for broadcasting
        t_enc = rearrange(t_enc, 't d -> 1 d t 1 1')
        t_enc = repeat(t_enc, '1 d t 1 1 -> 1 d t h w', h=H, w=W)

        return t_enc


class SpectralConv3d(nn.Module):
    """3D Spectral convolution for space-time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_spatial: int,
        modes_temporal: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_spatial = modes_spatial
        self.modes_temporal = modes_temporal

        # Complex weights
        self.scale = 1 / (in_channels * out_channels)
        self.weights_real = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels,
                modes_temporal, modes_spatial, modes_spatial
            )
        )
        self.weights_imag = nn.Parameter(
            self.scale * torch.randn(
                in_channels, out_channels,
                modes_temporal, modes_spatial, modes_spatial
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through 3D spectral convolution.

        Args:
            x: Input tensor of shape [B, C, T, H, W]

        Returns:
            Output tensor of shape [B, C', T, H, W]
        """
        B, C, T, H, W = x.shape

        # Compute 3D FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1], norm='ortho')

        # Truncate to relevant modes
        x_ft = x_ft[..., :self.modes_temporal, :self.modes_spatial, :self.modes_spatial]

        # Complex multiplication
        weights = torch.complex(self.weights_real, self.weights_imag)
        out_ft = torch.einsum('bctxy,cdtxy->bdtxy', x_ft, weights)

        # Pad back
        out_ft_padded = torch.zeros(
            B, self.out_channels, T, H, W // 2 + 1,
            dtype=torch.complex64, device=x.device
        )
        out_ft_padded[
            ..., :self.modes_temporal, :self.modes_spatial, :self.modes_spatial
        ] = out_ft

        # Inverse FFT
        out = torch.fft.irfftn(out_ft_padded, s=(T, H, W), dim=[-3, -2, -1], norm='ortho')

        return out