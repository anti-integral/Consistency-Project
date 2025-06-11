import torch

class EMA:
    """
    Exponential Moving Average of model parameters.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def store(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model):
        model.load_state_dict(self.backup, strict=False)
        del self.backup
