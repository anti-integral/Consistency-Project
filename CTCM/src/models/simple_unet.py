"""Simplified UNet that works correctly."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .time_conditioning import PositionalEmbedding, AdaptiveGroupNorm


class SimpleResBlock(nn.Module):
    """Simple residual block."""

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels * 2)
        )

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.norm1(h)

        # Add time embedding
        time_out = self.time_mlp(time_emb)
        scale, shift = torch.chunk(time_out, 2, dim=1)
        h = h * (1 + scale.unsqueeze(-1).unsqueeze(-1))
        h = h + shift.unsqueeze(-1).unsqueeze(-1)

        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.skip(x)


class SimpleUNetModel(nn.Module):
    """Simplified UNet that avoids dimension mismatches."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 64,
        channel_mult: List[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        **kwargs  # Ignore other args for compatibility
    ):
        super().__init__()

        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult

        # Time embedding
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()

        ch = channels
        for level, mult in enumerate(channel_mult):
            # Residual blocks at this resolution
            for i in range(num_res_blocks):
                self.down_blocks.append(
                    SimpleResBlock(ch, channels * mult, time_embed_dim)
                )
                ch = channels * mult

            # Downsample (except at the last level)
            if level != len(channel_mult) - 1:
                self.down_samples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            SimpleResBlock(ch, ch, time_embed_dim),
            SimpleResBlock(ch, ch, time_embed_dim),
        ])

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            # Upsample (except at the first iteration)
            if level != len(channel_mult) - 1:
                self.up_samples.append(
                    nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)
                )

            # Residual blocks at this resolution
            for i in range(num_res_blocks + 1):
                in_ch = ch + channels * mult if i == 0 else channels * mult
                self.up_blocks.append(
                    SimpleResBlock(in_ch, channels * mult, time_embed_dim)
                )
            ch = channels * mult

        # Output layers
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_embed(t)

        # Initial conv
        h = self.conv_in(x)

        # Downsampling
        hs = []

        # Process each resolution level
        block_idx = 0
        for level, mult in enumerate(self.channel_mult):
            # Process residual blocks at this resolution
            for i in range(self.num_res_blocks):
                h = self.down_blocks[block_idx](h, time_emb)
                block_idx += 1

            # Save skip connection
            hs.append(h)

            # Downsample (except at the last level)
            if level != len(self.channel_mult) - 1:
                h = self.down_samples[level](h)

        # Middle
        for block in self.middle_blocks:
            h = block(h, time_emb)

        # Upsampling
        block_idx = 0
        for level, mult in reversed(list(enumerate(self.channel_mult))):
            # Upsample (except at the first iteration)
            if level != len(self.channel_mult) - 1:
                h = self.up_samples[len(self.channel_mult) - level - 2](h)

            # Concatenate skip connection
            h = torch.cat([h, hs.pop()], dim=1)

            # Process residual blocks at this resolution
            for i in range(self.num_res_blocks + 1):
                h = self.up_blocks[block_idx](h, time_emb)
                block_idx += 1

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h