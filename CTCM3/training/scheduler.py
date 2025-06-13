import torch
import math

class KarrasSchedule:
    """
    σ schedule from Karras et al. (EDM).  Useful for both training sampler and
    consistency‑model conditioning.
    """
    def __init__(self, sigma_min: float, sigma_max: float, rho: float):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        t ∈ [0,1] → σ(t) in log space.
        """
        rho = self.rho
        sigma = (self.sigma_max ** (1 / rho) + t * (self.sigma_min ** (1 / rho) - self.sigma_max ** (1 / rho))) ** rho
        return sigma
