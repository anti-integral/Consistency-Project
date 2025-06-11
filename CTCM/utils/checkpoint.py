import torch, os, pathlib

def save_ckpt(model, ema, optim, step, outdir, is_best=False):
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    obj = {
        "model": model.state_dict(),
        "ema":   ema.shadow,
        "optim": optim.state_dict(),
        "step":  step,
    }
    torch.save(obj, os.path.join(outdir, f"step_{step}.pt"))
    if is_best:
        torch.save(obj, os.path.join(outdir, "best_ema.pt"))

def load_ckpt(path, model, ema, optim=None):
    obj = torch.load(path, map_location="cpu")
    model.load_state_dict(obj["model"], strict=False)
    ema.shadow = obj["ema"]
    if optim:
        optim.load_state_dict(obj["optim"])
    return obj["step"]
