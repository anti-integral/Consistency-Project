"""Time conditioning modules for stable consistency model training.

Implements TrigFlow-style positional embeddings and adaptive normalization.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Positional embedding for time conditioning.

    Uses sinusoidal embeddings instead of Fourier features for stability.
    """

    def __init__(
        self,
        embed_dim: int,
        max_positions: int = 10000,
        endpoint: bool = False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.max_positions = max_positions
        self.endpoint = endpoint

        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_positions) *
            torch.arange(half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate positional embeddings for time values.

        Args:
            t: Time values of shape [B] or [B, 1]

        Returns:
            Embeddings of shape [B, embed_dim]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        # Apply frequency bands
        args = t * self.freqs.unsqueeze(0)

        # Generate sin/cos embeddings
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        # Handle odd embedding dimension
        if self.embed_dim % 2:
            embedding = torch.cat([
                embedding,
                torch.zeros_like(embedding[:, :1])
            ], dim=-1)

        return embedding


class AdaptiveGroupNorm(nn.Module):
    """Adaptive Group Normalization with optional PixelNorm.

    Incorporates time embeddings via scale and shift parameters.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        time_embed_dim: int,
        use_scale_shift: bool = True,
        use_pixel_norm: bool = True,
        eps: float = 1e-5,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.use_scale_shift = use_scale_shift
        self.use_pixel_norm = use_pixel_norm
        self.eps = eps

        # Group normalization
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)

        # Time embedding projection
        if use_scale_shift:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, 2 * num_channels)
            )
        else:
            self.time_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_embed_dim, num_channels)
            )

        # Initialize projection to zero for stability
        with torch.no_grad():
            self.time_proj[-1].weight.zero_()
            self.time_proj[-1].bias.zero_()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Apply adaptive group normalization.

        Args:
            x: Input tensor of shape [B, C, H, W]
            time_emb: Time embeddings of shape [B, time_embed_dim]

        Returns:
            Normalized tensor of shape [B, C, H, W]
        """
        # Apply group normalization
        h = self.norm(x)

        # Get scale and shift from time embeddings
        time_out = self.time_proj(time_emb)

        if self.use_scale_shift:
            scale, shift = torch.chunk(time_out, 2, dim=1)
            h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1))
            h = h + shift.unsqueeze(-1).unsqueeze(-1)
        else:
            h = h + time_out.unsqueeze(-1).unsqueeze(-1)

        # Apply pixel normalization if enabled
        if self.use_pixel_norm:
            h = h * torch.rsqrt(
                torch.mean(h**2, dim=1, keepdim=True) + self.eps
            )

        return h


class TimestepEmbedding(nn.Module):
    """Learned timestep embedding with optional positional encoding."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        use_positional: bool = True,
        max_period: float = 10000.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim * 4
        self.use_positional = use_positional

        if use_positional:
            self.positional = PositionalEmbedding(embed_dim, max_period)

        # MLP to process embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate timestep embeddings.

        Args:
            t: Timesteps of shape [B]

        Returns:
            Embeddings of shape [B, embed_dim]
        """
        if self.use_positional:
            emb = self.positional(t)
        else:
            # Simple learnable embedding
            emb = t.unsqueeze(-1) * torch.ones(
                1, self.embed_dim, device=t.device
            )

        # Process through MLP
        emb = self.mlp(emb)

        return emb


class TrigFlowTimeConditioning(nn.Module):
    """Complete time conditioning module with TrigFlow improvements."""

    def __init__(
        self,
        base_channels: int,
        out_channels: int,
        use_positional: bool = True,
        use_scale_shift: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.use_positional = use_positional
        self.use_scale_shift = use_scale_shift

        # Time embedding
        self.time_embed = TimestepEmbedding(
            base_channels,
            hidden_dim=out_channels,
            use_positional=use_positional,
        )

        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Output projection
        self.out_proj = nn.Sequential(
            nn.Linear(base_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Generate time conditioning features.

        Args:
            t: Time values of shape [B]

        Returns:
            Time features of shape [B, out_channels]
        """
        # Get base embeddings
        emb = self.time_embed(t)

        # Apply dropout
        emb = self.dropout(emb)

        # Project to output dimension
        out = self.out_proj(emb)

        return out