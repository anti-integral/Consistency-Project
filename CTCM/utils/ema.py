# utils/ema.py

import torch
import copy

class EMA:
    """
    Exponential Moving Average of model parameters,
    safe against non-Tensor state_dict entries.
    """
    def __init__(self, model, decay: float):
        self.decay = decay
        self.shadow = {}
        # Clone Tensors, deepcopy everything else
        for k, v in model.state_dict().items():
            if isinstance(v, torch.Tensor):
                self.shadow[k] = v.clone().detach()
            else:
                self.shadow[k] = copy.deepcopy(v)

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if isinstance(v, torch.Tensor):
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
            else:
                # non-Tensor entries left as-is
                self.shadow[k] = copy.deepcopy(v)

    def store(self, model):
        # backup current params
        self.backup = {}
        for k, v in model.state_dict().items():
            self.backup[k] = v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        # load EMA params
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        # restore backup
        model.load_state_dict(self.backup, strict=False)
        del self.backup
