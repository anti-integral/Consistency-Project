"""U-Net architecture optimized for consistency models.

Incorporates TrigFlow-style improvements and neural operator layers.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .time_conditioning import AdaptiveGroupNorm, PositionalEmbedding
from .neural_operators import FourierConvBlock


class ResBlock(nn.Module):
    """Residual block with adaptive group normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
        num_groups: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_scale_shift_norm = use_scale_shift_norm

        # Ensure num_groups is valid
        num_groups = min(num_groups, min(in_channels, out_channels))

        self.norm1 = AdaptiveGroupNorm(
            num_groups, in_channels, time_embed_dim,
            use_scale_shift_norm, use_pixel_norm=False, eps=1e-5
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = AdaptiveGroupNorm(
            num_groups, out_channels, time_embed_dim,
            use_scale_shift_norm, use_pixel_norm=False, eps=1e-5
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x, time_emb)
        h = F.silu(h)
        h = self.conv1(h)

        h = self.norm2(h, time_emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


class AttentionBlock(nn.Module):
    """Multi-head self-attention block."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 1,
        num_head_channels: int = -1,
        use_checkpoint: bool = False,
    ):
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        if num_head_channels == -1:
            self.num_head_channels = channels // num_heads
        else:
            self.num_head_channels = num_head_channels

        # Ensure num_groups is valid
        num_groups = min(32, channels)
        self.norm = nn.GroupNorm(num_groups, channels, eps=1e-5)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.num_heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.num_heads)

        # Scaled dot-product attention
        scale = self.num_head_channels ** -0.5
        attn = torch.einsum('b h c i, b h c j -> b h i j', q, k) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.einsum('b h i j, b h c j -> b h c i', attn, v)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', x=H, y=W)

        return x + self.proj_out(out)


class Downsample(nn.Module):
    """Downsampling layer."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, 2)


class Upsample(nn.Module):
    """Upsampling layer."""

    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv

        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x


class UNetModel(nn.Module):
    """U-Net model with neural operator enhancements."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: int = 128,
        channel_mult: List[int] = [1, 2, 2, 2],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16, 8],
        dropout: float = 0.0,
        use_checkpoint: bool = False,
        num_heads: int = 4,
        num_head_channels: int = 64,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = True,
        use_fno: bool = True,
        fno_modes: int = 16,
        time_embed_dim: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.use_checkpoint = use_checkpoint
        self.use_fno = use_fno

        # Time embedding
        self.time_embed = nn.Sequential(
            PositionalEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1)

        # Track channels for skip connections
        ch = channels
        ds = 1
        
        # Create downsampling blocks
        self.down_blocks = nn.ModuleList()
        # Store (module, is_downsample) tuples to track which are downsample layers
        down_block_types = []
        
        for level, mult in enumerate(channel_mult):
            for i in range(num_res_blocks):
                # ResBlock
                out_ch = mult * channels
                block = ResBlock(
                    ch, out_ch, time_embed_dim,
                    dropout, use_scale_shift_norm
                )
                self.down_blocks.append(block)
                down_block_types.append(('res', ch))
                ch = out_ch

                # Attention
                if ds in attention_resolutions:
                    att_block = AttentionBlock(
                        ch, num_heads, num_head_channels, use_checkpoint
                    )
                    self.down_blocks.append(att_block)
                    down_block_types.append(('attn', ch))

                # FNO
                if use_fno and i == 0:
                    fno_block = FourierConvBlock(ch, ch, fno_modes)
                    self.down_blocks.append(fno_block)
                    down_block_types.append(('fno', ch))

            # Downsample
            if level != len(channel_mult) - 1:
                downsample = Downsample(ch, use_conv=resblock_updown)
                self.down_blocks.append(downsample)
                down_block_types.append(('down', ch))
                ds *= 2

        # Middle blocks
        self.middle_blocks = nn.ModuleList([
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
            AttentionBlock(ch, num_heads, num_head_channels, use_checkpoint),
            ResBlock(ch, ch, time_embed_dim, dropout, use_scale_shift_norm),
        ])

        # Create upsampling blocks
        self.up_blocks = nn.ModuleList()
        # We need to reverse the order and match skip connections properly
        
        # Calculate skip channels by reversing the downsampling path
        skip_channels = []
        for block_type, block_ch in down_block_types:
            if block_type != 'down':  # Don't store skip for downsample layers
                skip_channels.append(block_ch)
        skip_channels = list(reversed(skip_channels))
        skip_idx = 0
        
        for level, mult in reversed(list(enumerate(channel_mult))):
            # Upsample first (except at the last level)
            if level != len(channel_mult) - 1:
                upsample = Upsample(ch, use_conv=resblock_updown)
                self.up_blocks.append(upsample)
                ds //= 2
            
            for i in range(num_res_blocks + 1):
                # Get skip channel count
                skip_ch = skip_channels[skip_idx] if skip_idx < len(skip_channels) else 0
                skip_idx += 1
                
                # ResBlock with skip
                out_ch = mult * channels
                block = ResBlock(
                    ch + skip_ch, out_ch, time_embed_dim,
                    dropout, use_scale_shift_norm
                )
                self.up_blocks.append(block)
                ch = out_ch

                # Skip connection for attention if it was in down path
                if ds in attention_resolutions and i < num_res_blocks:
                    if skip_idx < len(skip_channels):
                        skip_idx += 1
                    att_block = AttentionBlock(
                        ch, num_heads, num_head_channels, use_checkpoint
                    )
                    self.up_blocks.append(att_block)

                # Skip connection for FNO if it was in down path
                if use_fno and i == num_res_blocks and level < len(channel_mult) - 1:
                    if skip_idx < len(skip_channels):
                        skip_idx += 1
                    fno_block = FourierConvBlock(ch, ch, fno_modes)
                    self.up_blocks.append(fno_block)

        # Output layers
        num_groups = min(32, ch)
        self.norm_out = nn.GroupNorm(num_groups, ch, eps=1e-5)
        self.conv_out = nn.Conv2d(ch, out_channels, 3, padding=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [B, C, H, W]
            t: Time tensor of shape [B]

        Returns:
            Output tensor of shape [B, C, H, W]
        """
        # Get time embeddings
        time_emb = self.time_embed(t)

        # Initial convolution
        h = self.conv_in(x)
        
        # Downsampling - store features before downsampling operations
        hs = []
        for module in self.down_blocks:
            if isinstance(module, ResBlock):
                h = module(h, time_emb)
                hs.append(h)  # Save for skip
            elif isinstance(module, (AttentionBlock, FourierConvBlock)):
                h = module(h)
                hs.append(h)  # Save for skip
            elif isinstance(module, Downsample):
                # Don't save downsampled features as skip
                h = module(h)

        # Middle
        for layer in self.middle_blocks:
            if isinstance(layer, ResBlock):
                h = layer(h, time_emb)
            else:
                h = layer(h)

        # Upsampling - pop skip connections in reverse order
        for module in self.up_blocks:
            if isinstance(module, Upsample):
                h = module(h)
            elif isinstance(module, ResBlock):
                # Concatenate skip connection
                if len(hs) > 0:
                    h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, time_emb)
            elif isinstance(module, (AttentionBlock, FourierConvBlock)):
                h = module(h)
                # Pop corresponding skip if it exists
                if len(hs) > 0 and isinstance(module, type(self.down_blocks[len(self.down_blocks) - len(hs) - 1])):
                    hs.pop()

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h