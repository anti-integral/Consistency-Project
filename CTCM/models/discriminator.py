import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    Lightweight patch discriminator (DCGAN‑style).
    Used only if GAN loss is enabled.
    """
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()
        c = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c*2, 4, 2, 1), nn.BatchNorm2d(c*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*2, c*4, 4, 2, 1), nn.BatchNorm2d(c*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c*4, 1, 4, 1, 0)  # (B,1,1,1)
        )

    def forward(self, x):
        return self.net(x).view(-1)
