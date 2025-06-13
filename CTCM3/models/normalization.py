import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaGN(nn.Module):
    """
    Adaptive GroupNorm with PixelNorm on the time embedding.
    Follows Lu & Song (2025) for continuous‑time consistency models.
    """
    def __init__(self, channels: int, embed_dim: int, num_groups: int = 32):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups=num_groups, num_channels=channels, eps=1e-6)
        self.proj = nn.Linear(embed_dim, channels * 2)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        # PixelNorm on t_emb
        t_emb = t_emb / torch.sqrt((t_emb ** 2).mean(dim=-1, keepdim=True) + 1e-8)
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)  # [B, C] each
        scale = scale[:, :, None, None]  # broadcast
        shift = shift[:, :, None, None]
        x = self.gn(x)
        return x * (1 + scale) + shift
