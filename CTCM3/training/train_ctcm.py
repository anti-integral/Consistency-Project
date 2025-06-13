import yaml, math, time, os, argparse
from pathlib import Path
from tqdm import tqdm

import torch
import torch.optim as optim

from datasets.cifar import get_cifar_loaders
from models import UNet
from training.loss import consistency_loss
from training.ema import EMA
from training.scheduler import KarrasSchedule
from utils.seed import seed_everything, save_checkpoint

# ----------------------------------------------------------------------------- #
# Argument parsing
# ----------------------------------------------------------------------------- #
def build_config(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main(cfg_path: str):
    cfg = build_config(cfg_path)
    seed_everything(42)

    # Data
    train_loader, _ = get_cifar_loaders(cfg["train"]["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = UNet(
        img_size=cfg["model"]["img_size"],
        in_channels=3,
        base_dim=cfg["model"]["base_dim"],
        dim_mults=cfg["model"]["dim_mults"],
        num_res_blocks=cfg["model"]["num_res_blocks"],
        time_embed_dim=cfg["model"]["time_embed_dim"],
        out_channels=3,
        dropout=cfg["model"]["dropout"],
    ).to(device)

    # EMA
    ema = EMA(model, decay=cfg["train"]["ema_decay"])

    # Optimiser
    opt = optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=(0.9, 0.99),
    )

    schedule = KarrasSchedule(
        sigma_min=cfg["train"]["sigma_min"],
        sigma_max=cfg["train"]["sigma_max"],
        rho=cfg["train"]["rho"],
    )

    step = 0
    pbar = tqdm(total=cfg["train"]["total_steps"])
    while step < cfg["train"]["total_steps"]:
        for x, _ in train_loader:
            x = x.to(device)
            loss = consistency_loss(
                model,
                x,
                schedule,
                sigma_data=cfg["train"]["sigma_data"],
                tangent_clip=cfg["train"]["tangent_clip"],
                warmup=cfg["train"]["warmup_steps"],
                step=step,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ema.update()

            step += 1
            pbar.update(1)
            pbar.set_description(f"step {step} loss {loss.item():.4f}")

            if step % cfg["train"]["save_every"] == 0 or step == cfg["train"]["total_steps"]:
                Path("ckpts").mkdir(exist_ok=True)
                save_checkpoint(model.state_dict(), f"ckpts/ctcm_step{step}.pt")
                save_checkpoint(ema.shadow,         f"ckpts/ctcm_ema_step{step}.pt")

            if step >= cfg["train"]["total_steps"]:
                break

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/cifar10_default.yaml")
    args = parser.parse_args()
    main(args.config)
