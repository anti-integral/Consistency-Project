import torchvision.utils as vutils
import torch

def save_image_grid(tensor, filename, nrow=8):
    """tensor: (B,C,H,W) in [-1,1]"""
    vutils.save_image(tensor, filename, nrow=nrow, normalize=True, value_range=(-1,1))
