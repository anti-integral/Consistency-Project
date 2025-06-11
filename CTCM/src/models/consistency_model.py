"""Neural Operator Continuous Time Consistency Model.

Implements TrigFlow parameterization with FNO trajectory learning.
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp, vjp
import numpy as np

from .unet import UNetModel
from .ema import ExponentialMovingAverage


class ConsistencyModel(nn.Module):
    """Continuous-time consistency model with neural operator backbone."""

    def __init__(
        self,
        backbone: nn.Module,
        sigma_data: float = 0.5,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
        parameterization: str = "v-prediction",
        use_trigflow: bool = True,
    ):
        super().__init__()

        self.backbone = backbone
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.parameterization = parameterization
        self.use_trigflow = use_trigflow

        # Precompute time schedule boundaries
        self.register_buffer(
            "t_min", torch.tensor(self._sigma_to_t(sigma_min))
        )
        self.register_buffer(
            "t_max", torch.tensor(self._sigma_to_t(sigma_max))
        )

    def _sigma_to_t(self, sigma: Union[float, torch.Tensor]) -> Union[float, torch.Tensor]:
        """Convert noise level to time using TrigFlow parameterization."""
        if self.use_trigflow:
            # TrigFlow: t = arctan(sigma / sigma_data)
            if isinstance(sigma, float):
                return float(torch.arctan(torch.tensor(sigma / self.sigma_data)))
            else:
                return torch.arctan(sigma / self.sigma_data)
        else:
            # Original: t = sigma
            return sigma

    def _t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert time to noise level."""
        if self.use_trigflow:
            # TrigFlow: sigma = sigma_data * tan(t)
            return self.sigma_data * torch.tan(t)
        else:
            # Original: sigma = t
            return t

    def get_scalings(self, sigma: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get scaling factors for different parameterizations."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / torch.sqrt(sigma**2 + self.sigma_data**2)

        return {
            "c_skip": c_skip,
            "c_out": c_out,
            "c_in": c_in,
        }

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        return_dict: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass of consistency model.

        Args:
            x: Noisy input at time t, shape [B, C, H, W]
            t: Time values, shape [B]
            return_dict: Whether to return additional info

        Returns:
            Predicted clean image or dict with predictions
        """
        # Ensure t is in valid range
        t = t.clamp(self.t_min, self.t_max)

        # Convert to sigma
        sigma = self._t_to_sigma(t)

        # Get scalings
        scalings = self.get_scalings(sigma.view(-1, 1, 1, 1))

        # Scale input
        x_scaled = scalings["c_in"] * x

        # Get model prediction
        model_output = self.backbone(x_scaled, t)

        # Apply skip connection and output scaling
        if self.parameterization == "v-prediction":
            # v-parameterization predicts v = (x - eps) / sqrt(sigma^2 + sigma_data^2)
            v_pred = model_output
            x_pred = scalings["c_skip"] * x + scalings["c_out"] * v_pred
        elif self.parameterization == "eps-prediction":
            # eps-parameterization predicts noise
            eps_pred = model_output
            x_pred = (x - sigma.view(-1, 1, 1, 1) * eps_pred) / torch.sqrt(
                1 + sigma.view(-1, 1, 1, 1)**2 / self.sigma_data**2
            )
        else:
            # Direct x0 prediction
            x_pred = scalings["c_skip"] * x + scalings["c_out"] * model_output

        if return_dict:
            return {
                "x_pred": x_pred,
                "model_output": model_output,
                "sigma": sigma,
                "scalings": scalings,
            }
        else:
            return x_pred

    def get_tangent(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute tangent vector for continuous-time consistency training.

        Uses automatic differentiation to compute dx/dt along the trajectory.
        """
        # Enable gradient computation
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # Forward pass
        x_pred = self.forward(x, t)

        # Compute dx/dt using autograd
        dx_dt = torch.autograd.grad(
            outputs=x_pred,
            inputs=t,
            grad_outputs=torch.ones_like(x_pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        return dx_dt

    def compute_consistency_loss(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_model: Optional[nn.Module] = None,
        loss_fn: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute continuous-time consistency loss.

        Args:
            x: Clean data, shape [B, C, H, W]
            t: Time values, shape [B]
            target_model: Target model for consistency (if None, uses self)
            loss_fn: Loss function (if None, uses L2)

        Returns:
            Consistency loss value
        """
        if target_model is None:
            target_model = self
        if loss_fn is None:
            loss_fn = nn.MSELoss()

        # Sample noise
        sigma = self._t_to_sigma(t)
        eps = torch.randn_like(x)
        x_t = x + sigma.view(-1, 1, 1, 1) * eps

        # Get predictions
        with torch.no_grad():
            # Target prediction at current time
            target_pred = target_model(x_t, t)

        # Student prediction
        student_pred = self.forward(x_t, t)

        # For continuous time, we need to match tangent vectors
        if hasattr(loss_fn, 'continuous_time') and loss_fn.continuous_time:
            # Compute tangent matching loss
            student_tangent = self.get_tangent(x_t, t)

            with torch.no_grad():
                target_tangent = target_model.get_tangent(x_t, t)

            # Tangent consistency loss
            loss = loss_fn(student_tangent, target_tangent)
        else:
            # Standard consistency loss
            loss = loss_fn(student_pred, target_pred)

        return loss

    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: int = 1,
        device: torch.device = torch.device("cuda"),
        guidance_scale: float = 0.0,
        class_labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using the consistency model.

        Args:
            shape: Shape of samples to generate [B, C, H, W]
            num_steps: Number of sampling steps
            device: Device to generate on
            guidance_scale: Classifier-free guidance scale
            class_labels: Optional class labels for conditional generation

        Returns:
            Generated samples
        """
        batch_size = shape[0]

        # Start from noise
        x = torch.randn(shape, device=device) * self.sigma_max

        # Create time schedule
        if num_steps == 1:
            # Single step: directly denoise from max noise
            t = torch.full((batch_size,), self.t_max, device=device)
            x = self.forward(x, t)
        else:
            # Multi-step sampling
            t_steps = torch.linspace(self.t_max, self.t_min, num_steps + 1, device=device)

            for i in range(num_steps):
                t_curr = t_steps[i]
                t_next = t_steps[i + 1]

                # Denoise to current time
                t_batch = torch.full((batch_size,), t_curr, device=device)
                x_0 = self.forward(x, t_batch)

                # Add noise to go to next time (except last step)
                if i < num_steps - 1:
                    sigma_next = self._t_to_sigma(t_next)
                    eps = torch.randn_like(x)
                    x = x_0 + sigma_next * eps
                else:
                    x = x_0

        return x

    def get_karras_sigmas(
        self,
        n: int,
        device: torch.device = torch.device("cuda"),
    ) -> torch.Tensor:
        """Get Karras noise schedule for progressive training."""
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho
        return sigmas