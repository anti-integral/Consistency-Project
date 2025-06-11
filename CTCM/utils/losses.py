import torch
import torch.nn.functional as F
from torch.func import jvp

def karras_sigma(t, sigma_min, sigma_max, rho=7.0):
    """
    Continuous log‑log schedule (Karras et al. 2022).
    """
    sigma = (sigma_max ** (1 / rho) + t * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    return sigma

def consistency_and_tangent_loss(model, x0, t, *, sigma_min, sigma_max, params, tangent_lambda):
    """
    Compute consistency MSE + JVP tangent regulariser.
    """
    device = x0.device
    # sample a second, slightly earlier time
    dt = torch.rand_like(t) * 0.06 + 1e-3    # Δt ∈ (0.001, 0.061)
    t2 = torch.clamp(t - dt, 0.0, 1.0)

    sigma1 = karras_sigma(t,  sigma_min, sigma_max)
    sigma2 = karras_sigma(t2, sigma_min, sigma_max)

    eps = torch.randn_like(x0)
    x_t1 = x0 + sigma1[:, None, None, None] * eps
    x_t2 = x0 + sigma2[:, None, None, None] * eps

    pred1 = model(x_t1, t)
    with torch.no_grad():
        pred2 = model(x_t2, t2)

    cons_loss = F.mse_loss(pred1, pred2)

    # Tangent loss (Jacobian‑vector product wrt input)
    def f(inp):
        return model(inp, t)

    v = (x_t2 - x_t1) / (sigma2 - sigma1)[:, None, None, None]
    _, jvp_out = jvp(f, (x_t1,), (v,))
    tan_loss = jvp_out.pow(2).mean()

    return cons_loss + tangent_lambda * tan_loss
