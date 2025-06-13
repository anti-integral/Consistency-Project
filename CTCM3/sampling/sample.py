import yaml, argparse, math, os
from pathlib import Path
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from cleanfid import fid

from models import UNet
from training.scheduler import KarrasSchedule
from utils.seed import seed_everything

@torch.no_grad()
def generate(cfg, ckpt_path, outdir="samples", batch_size=100):
    seed_everything(0)
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
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    schedule = KarrasSchedule(
        sigma_min=cfg["train"]["sigma_min"],
        sigma_max=cfg["train"]["sigma_max"],
        rho=cfg["train"]["rho"],
    )

    n_samples = cfg["eval"]["n_samples"]
    Path(outdir).mkdir(exist_ok=True)
    n_iter = math.ceil(n_samples / batch_size)
    all_imgs = []

    for i in tqdm(range(n_iter)):
        cur_bs = min(batch_size, n_samples - i * batch_size)
        # Sample pure noise
        x = torch.randn(cur_bs, 3, cfg["model"]["img_size"], cfg["model"]["img_size"], device=device)
        # One‑step generation (t = 1 → 0)
        sigma = torch.full((cur_bs,), schedule.sigma(torch.ones(()).to(device)), device=device)
        c_in = 1.0 / torch.sqrt(sigma**2 + cfg["train"]["sigma_data"]**2)
        x = x / sigma.view(cur_bs, 1, 1, 1)  # normalise noise
        pred = model(x * c_in.view(cur_bs,1,1,1), sigma)
        all_imgs.append(pred.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)[:n_samples]
    # Un‑normalise back to [0,1] for FID
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
    std  = torch.tensor([0.2470, 0.2435, 0.2616]).view(1,3,1,1)
    imgs = torch.clamp(all_imgs * std + mean, 0, 1)

    # Save individual PNGs (Clean‑FID requirement)
    for idx, img in enumerate(imgs):
        save_image(img, f"{outdir}/{idx:05d}.png")

    print("Computing FID…")
    score = fid.compute_fid(
        fdir1=outdir,
        dataset_name="cifar10",
        dataset_split="test",
        device=cfg["eval"]["fid_devices"],
    )
    print(f"FID = {score:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, default="configs/cifar10_default.yaml")
    parser.add_argument("--ckpt",    type=str, required=True)
    parser.add_argument("--outdir",  type=str, default="samples")
    parser.add_argument("--bs",      type=int, default=100)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    generate(cfg, args.ckpt, args.outdir, args.bs)
