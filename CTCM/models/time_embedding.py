import math
import torch
import torch.nn as nn

class SinusoidalTimeEmbedding(nn.Module):
    """
    Low‑frequency sinusoidal embedding as recommended by Lu & Song (2024).
    """
    def __init__(self, embed_dim: int = 128, max_period: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t : shape (B,) float32 in [0,1]
        Returns embedding (B, embed_dim)
        """
        device = t.device
        half_dim = self.embed_dim // 2
        freqs = torch.arange(half_dim, device=device) / half_dim
        freqs = self.max_period ** freqs
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb
