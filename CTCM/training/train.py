import torch, itertools, math
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset import get_cifar_dataloader
from models.fno_generator import FNOGenerator
from utils.ema import EMA
from utils.losses import consistency_and_tangent_loss
from utils import checkpoint as ckpt_utils
from utils.config import load_config

def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FNOGenerator(**cfg["fno"]).to(device)
    ema = EMA(model, cfg["ema_decay"])
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    loader = get_cifar_dataloader(cfg["batch_size"], cfg["image_size"])

    start_step = 0
    if cfg["resume"]:
        start_step = ckpt_utils.load_ckpt(cfg["resume"], model, ema, optim)

    total_steps = cfg["num_epochs"] * len(loader)
    global_step = start_step
    best_fid = math.inf

    for epoch in range(cfg["num_epochs"]):
        for x, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            x = x.to(device, non_blocking=True)
            t = torch.rand(x.size(0), device=device)

            loss = consistency_and_tangent_loss(
                model, x, t,
                sigma_min=cfg["sigma_min"],
                sigma_max=cfg["sigma_max"],
                params=None,
                tangent_lambda=cfg["tangent_lambda"],
            )

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ema.update(model)

            if global_step % cfg["save_interval"] == 0 and global_step > 0:
                ckpt_utils.save_ckpt(model, ema, optim, global_step, "ckpt")
            global_step += 1

if __name__ == "__main__":
    main()
