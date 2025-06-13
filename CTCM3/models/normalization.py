# models/normalization.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    """
    Adaptive GroupNorm: normalize input x, then inject time embedding
    as per-group scale & shift.

    Args:
        channels  (int): number of feature channels in x
        embed_dim (int): dimensionality of the time embedding t_emb
        num_groups (int): desired number of groups for GroupNorm
    """
    def __init__(self, channels: int, embed_dim: int, num_groups: int = 32):
        super().__init__()
        # Ensure groups divide channels
        self.num_groups = min(num_groups, channels)
        # GroupNorm over actual groups
        self.gn   = nn.GroupNorm(num_groups=self.num_groups, num_channels=channels, eps=1e-6)
        # Project time embed to 2 * G parameters (gamma & beta per group)
        self.proj = nn.Linear(embed_dim, self.num_groups * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     [B, C, H, W]  - feature map
        t_emb: [B, embed_dim] - time embedding
        """
        b, c, h, w = x.shape

        # 1) Normalize
        x_norm = self.gn(x)

        # 2) Project time embedding to per-group scale & shift
        #    yields [B, 2*G]
        gamma_beta = self.proj(t_emb)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, G]

        # 3) Reshape for broadcasting: [B, G, 1, 1]
        gamma = gamma.view(b, self.num_groups, 1, 1)
        beta  = beta.view(b, self.num_groups, 1, 1)

        # 4) Split channels into groups: [B, G, C/G, H, W]
        xg = x_norm.view(b, self.num_groups, c // self.num_groups, h, w)

        # 5) Apply scale & shift per group
        xg = xg * (1 + gamma.unsqueeze(2)) + beta.unsqueeze(2)

        # 6) Restore shape [B, C, H, W]
        return xg.view(b, c, h, w)
