# training/train.py

import os
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Enable TF32 on H100/H200 for faster matmuls
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # optimize kernels for fixed input shapes

from data.dataset import get_cifar_dataloader
from models.fno_generator import FNOGenerator
from utils.ema import EMA
from utils.losses import consistency_and_tangent_loss
from utils.checkpoint import save_ckpt, load_ckpt
from utils.config import load_config

def main():
    cfg = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data loader with prefetch for speed
    train_loader = get_cifar_dataloader(
        batch_size=cfg["batch_size"],
        img_size=cfg["image_size"],
        num_workers=8,
    )

    # Model setup
    model = FNOGenerator(**cfg["fno"]).to(device, memory_format=torch.channels_last)
    # Compile model for graph optimization on PyTorch 2.x
    model = torch.compile(model, backend="inductor")

    # EMA for stability
    ema = EMA(model, cfg["ema_decay"])

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        amsgrad=True
    )

    # Mixed‐precision scaler
    scaler = GradScaler()

    # Resume if requested
    start_step = 0
    if cfg.get("resume"):
        start_step = load_ckpt(cfg["resume"], model, ema, optimizer)

    total_steps = cfg["num_epochs"] * len(train_loader)
    global_step = start_step

    # Training loop
    for epoch in range(cfg["num_epochs"]):
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{cfg['num_epochs']}]")
        for x, _ in pbar:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            t = torch.rand(x.size(0), device=device)

            # Forward + loss under autocast for speed
            with autocast():
                loss = consistency_and_tangent_loss(
                    model, x, t,
                    sigma_min=cfg["sigma_min"],
                    sigma_max=cfg["sigma_max"],
                    params=None,
                    tangent_lambda=cfg["tangent_lambda"],
                )

            # Backpropagate with GradScaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            # Optional gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA
            ema.update(model)

            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "step": global_step})

            # Save checkpoints periodically
            if global_step % cfg["save_interval"] == 0:
                save_ckpt(model, ema, optimizer, global_step, outdir="ckpt")

    # At end of training, save final EMA weights
    save_ckpt(model, ema, optimizer, global_step, outdir="ckpt", is_best=True)
    print("Training complete. Best EMA checkpoint saved to ckpt/best_ema.pt")


if __name__ == "__main__":
    main()
