import torch, argparse, os
from models.fno_generator import FNOGenerator
from utils.ema import EMA
from utils.visualize import save_image_grid
from utils.losses import karras_sigma

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=50000)
    ap.add_argument("--out_dir", type=str, default="samples")
    ap.add_argument("--two_step", action="store_true",
                    help="If set, use 2‑step sampler (improves FID slightly).")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("cfg", None)  # stored config (optional)
    model = FNOGenerator(**cfg["fno"]).cuda().eval()
    model.load_state_dict(ckpt["model"], strict=False)
    ema = EMA(model, 0.0)
    ema.shadow = ckpt["ema"]
    ema.store(model)   # swap to EMA params

    batch = 256
    images = []
    steps = 2 if args.two_step else 1
    t_vals = torch.tensor([1.0, 0.0], device="cuda")[:steps+1]

    for i in range(0, args.num, batch):
        n = min(batch, args.num - i)
        x = torch.randn(n, 3, 32, 32, device="cuda")  # start at t=1
        for j in range(steps):
            t = t_vals[j].expand(n)
            x = model(x, t)
            if steps == 2 and j == 0:
                # move to intermediate sigma (sqrt schedule)
                x = x + 0.1 * torch.randn_like(x)
        images.append(x.clamp(-1,1).cpu())

    ema.restore(model)
    images = torch.cat(images)[:args.num]
    save_image_grid(images[:64], os.path.join(args.out_dir, "preview.png"))
    torch.save(images, os.path.join(args.out_dir, "images.pt"))
    print("Sampling complete →", args.out_dir)

if __name__ == "__main__":
    main()
