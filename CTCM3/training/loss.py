import torch
import torch.nn.functional as F
from torch.func import jvp

def consistency_loss(
    model,
    x0,
    schedule,
    *,
    sigma_data: float,
    tangent_clip: float,
    warmup: float,
    step: int,
):
    """
    Compute Lu & Song (2025) continuous‑time consistency loss with
    tangent regularisation and adaptive clipping.
    """
    b = x0.size(0)
    device = x0.device

    # sample uniform t ∈ (0,1); choose Δ uniformly in small window
    t = torch.rand(b, device=device)
    dt = torch.rand(b, device=device) * 0.1  # width 0.1

    sigma_t     = schedule.sigma(t)
    sigma_tnext = schedule.sigma(t - dt)

    # Add noise
    eps = torch.randn_like(x0)
    xt     = x0 + sigma_t.view(b,1,1,1) * eps
    xtnext = x0 + sigma_tnext.view(b,1,1,1) * eps

    # Pre‑conditioning scalings (Karras)
    c_in = 1.0 / torch.sqrt(sigma_t**2 + sigma_data**2)
    xt_scaled = xt * c_in.view(b,1,1,1)

    # Forward
    def f(inp):
        return model(inp, sigma_t)

    pred_xt = f(xt_scaled)
    with torch.no_grad():
        c_in_next = 1.0 / torch.sqrt(sigma_tnext**2 + sigma_data**2)
        target = model(xtnext * c_in_next.view(b,1,1,1), sigma_tnext)

    # Main consistency loss
    loss_consistency = F.mse_loss(pred_xt, target)

    # Tangent regularisation (Jacobian‑vector product)
    tangent = (xtnext - xt) / (sigma_tnext - sigma_t).view(b,1,1,1)
    _, jvp_out = jvp(f, (xt_scaled,), (tangent * c_in.view(b,1,1,1),))
    tangent_loss = (jvp_out ** 2).mean()

    # Warmup & clip
    alpha = min(1.0, step / warmup)  # gradually introduce tangent term
    tangent_loss = torch.clamp(tangent_loss, 0.0, tangent_clip)

    return loss_consistency + 0.1 * alpha * tangent_loss
