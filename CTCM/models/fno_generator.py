import torch
import torch.nn as nn
from einops import rearrange
from neuraloperator.layers.fno import FNO2d
from .time_embedding import SinusoidalTimeEmbedding

class FiLM(nn.Module):
    """Feature‑wise Linear Modulation block for conditioning."""
    def __init__(self, in_channels: int, time_dim: int):
        super().__init__()
        self.linear = nn.Linear(time_dim, in_channels * 2)

    def forward(self, x, t_emb):
        gamma, beta = self.linear(t_emb).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return x * (1 + gamma) + beta

class SkipConv(nn.Module):
    """Local detail branch to mitigate FNO spectral bias."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

class FNOGenerator(nn.Module):
    """
    FNO‑backbone continuous‑time consistency network.
    Input: noisy image (B,C,H,W) in [-1,1], scalar t in [0,1].
    Output: predicted clean image.
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        width=128,
        modes=16,
        layers=6,
        time_dim=128,
        use_lowpass=True,
        skip_conv=True,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )
        self.fno = FNO2d(
            in_channels + 1,          # +1 time‑channel
            out_channels,
            n_modes=(modes, modes),
            hidden_channels=width,
            num_layers=layers,
            use_lowpass_filter=use_lowpass,
        )
        self.film = FiLM(out_channels, time_dim)
        self.skip_conv = SkipConv(out_channels) if skip_conv else None

    def forward(self, x, t):
        """
        x: (B,C,H,W)  t: (B,) float32
        """
        # prepare conditioning
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)

        t_channel = t[:, None, None, None].repeat(1, 1, x.shape[2], x.shape[3])
        x_in = torch.cat([x, t_channel], dim=1)

        y = self.fno(x_in)           # global spectral mixing
        y = self.film(y, t_emb)      # FiLM modulate
        if self.skip_conv:
            y = y + self.skip_conv(y)
        return torch.tanh(y)         # keep image in [-1,1]
