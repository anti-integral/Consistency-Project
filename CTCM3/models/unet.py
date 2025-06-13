from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .time_embed import SinusoidalPosEmb
from .normalization import AdaGN

# ----------------------------------------------------------------------------- #
# Blocks
# ----------------------------------------------------------------------------- #

def conv3x3(in_ch, out_ch):  # small helper
    return nn.Conv2d(in_ch, out_ch, 3, padding=1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim, dropout):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.norm1 = AdaGN(in_ch,  time_embed_dim)
        self.norm2 = AdaGN(out_ch, time_embed_dim)
        self.act   = nn.SiLU(inplace=True)
        self.drop  = nn.Dropout(dropout)
        self.skip  = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x, t_emb))
        h = self.conv1(h)
        h = self.act(self.norm2(h, t_emb))
        h = self.drop(h)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)

    def forward(self, x, t_emb):
        return self.op(x)

# ----------------------------------------------------------------------------- #
# UNet
# ----------------------------------------------------------------------------- #

class UNet(nn.Module):
    def __init__(
        self,
        img_size: int,
        in_channels: int,
        base_dim: int,
        dim_mults: list[int],
        num_res_blocks: int,
        time_embed_dim: int,
        out_channels: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.SiLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )

        # Initial conv
        self.init_conv = conv3x3(in_channels, base_dim)

        dims = [base_dim, *map(lambda m: base_dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # Downsampling blocks
        self.downs = nn.ModuleList()
        for i, (in_ch, out_ch) in enumerate(in_out):
            for _ in range(num_res_blocks):
                self.downs.append(ResBlock(in_ch, in_ch, time_embed_dim, dropout))
            self.downs.append(ResBlock(in_ch, out_ch, time_embed_dim, dropout))
            if i != len(in_out) - 1:
                self.downs.append(Downsample(out_ch))

        # Bottleneck
        mid_ch = dims[-1]
        self.mid = nn.Sequential(
            ResBlock(mid_ch, mid_ch, time_embed_dim, dropout),
            ResBlock(mid_ch, mid_ch, time_embed_dim, dropout),
        )

        # Upsampling blocks
        self.ups = nn.ModuleList()
        for i, (in_ch, out_ch) in reversed(list(enumerate(in_out))):
            if i != len(in_out) - 1:
                self.ups.append(Upsample(in_ch * 2))
            self.ups.append(ResBlock(in_ch * 2, in_ch, time_embed_dim, dropout))
            for _ in range(num_res_blocks):
                self.ups.append(ResBlock(in_ch, in_ch, time_embed_dim, dropout))

        # Final conv
        self.final_block = nn.Sequential(
            AdaGN(base_dim, time_embed_dim),
            nn.SiLU(),
            nn.Conv2d(base_dim, out_channels, 3, padding=1),
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, sigma: torch.Tensor):
        """
        x:      [B, 3, H, W]      noisy image
        sigma:  [B] or [B, 1]     noise level (std dev) – we embed log‑sigma
        """
        t_emb = self.time_mlp(sigma.log())  # [B, time_embed_dim]

        h = self.init_conv(x)
        residuals = [h]

        # Down
        for mod in self.downs:
            h = mod(h, t_emb) if isinstance(mod, ResBlock) else mod(h, t_emb)
            residuals.append(h)

        # Mid
        h = self.mid(h, t_emb) if isinstance(self.mid, ResBlock) else self.mid(h, t_emb)

        # Up
        for mod in self.ups:
            if isinstance(mod, Upsample):
                h = mod(h, t_emb)
            else:
                res = residuals.pop()
                h = torch.cat([h, res], dim=1)
                h = mod(h, t_emb)

        # Final
        return self.final_block(h)
