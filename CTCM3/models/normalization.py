# models/normalization.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    """
    Adaptive GroupNorm: normalizes activations via GroupNorm, then
    injects time embeddings as per-channel scale & shift.
    Handles any input channel size dynamically.
    """
    def __init__(self, embed_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        # We'll project the time embedding to 2 * num_groups values:
        # one scale and one shift per group
        self.proj = nn.Linear(embed_dim, num_groups * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:      [B, C, H, W]
        t_emb:  [B, embed_dim]
        """
        b, c, h, w = x.shape
        # Ensure groups divide channels
        g = min(self.num_groups, c)
        if c % g != 0:
            # fallback to LayerNorm if channel count isn't divisible
            x_norm = F.layer_norm(x, (c, h, w), eps=1e-6)
        else:
            x_norm = F.group_norm(x, g, eps=1e-6)

        # Project time embedding to per-group gamma & beta
        gamma_beta = self.proj(t_emb)           # [B, 2*g]
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # each [B, g]

        # Reshape for broadcasting over channels
        # First expand to per-channel: each group covers (c//g) channels
        gamma = gamma.view(b, g, 1, 1)           # [B, g, 1, 1]
        beta  = beta.view(b, g, 1, 1)

        # Reshape x_norm to [B, g, C//g, H, W], apply per-group scale & shift,
        # then reshape back to [B, C, H, W]
        xg = x_norm.view(b, g, c // g, h, w)
        xg = xg * (1 + gamma.unsqueeze(2)) + beta.unsqueeze(2)
        return xg.reshape(b, c, h, w)
