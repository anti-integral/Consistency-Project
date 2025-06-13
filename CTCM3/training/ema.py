import torch

class EMA:
    """Simple exponential moving average of parameters."""
    def __init__(self, model, decay):
        self.decay = decay
        self.model = model
        self.shadow = {k: p.clone().detach() for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self):
        for k, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[k].lerp_(p, 1.0 - self.decay)

    def copy_to(self, model):
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.data.copy_(self.shadow[k])
