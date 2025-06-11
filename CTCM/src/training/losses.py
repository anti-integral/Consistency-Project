"""Loss functions for consistency model training.

Implements Pseudo-Huber loss and continuous-time consistency losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


class PseudoHuberLoss(nn.Module):
    """Pseudo-Huber loss for robust consistency training.

    L(x, y) = sqrt(||x - y||^2 + c^2) - c

    This loss is more robust than L2 and avoids the evaluation bias
    of LPIPS while providing smooth gradients.
    """

    def __init__(self, c: float = 0.01, reduction: str = 'mean'):
        super().__init__()
        self.c = c
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Pseudo-Huber loss.

        Args:
            input: Predicted tensor
            target: Target tensor

        Returns:
            Loss value
        """
        diff = input - target
        loss = torch.sqrt((diff ** 2).sum(dim=(1, 2, 3)) + self.c ** 2) - self.c

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedPseudoHuberLoss(PseudoHuberLoss):
    """Weighted Pseudo-Huber loss with SNR-based weighting."""

    def __init__(
        self,
        c: float = 0.01,
        reduction: str = 'mean',
        sigma_data: float = 0.5,
    ):
        super().__init__(c, reduction)
        self.sigma_data = sigma_data

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute weighted Pseudo-Huber loss.

        Args:
            input: Predicted tensor
            target: Target tensor
            sigma: Noise levels for weighting

        Returns:
            Weighted loss value
        """
        # Base loss
        loss = super().forward(input, target)

        # SNR weighting
        weight = 1 / (sigma ** 2 + self.sigma_data ** 2)

        if self.reduction == 'none':
            return loss * weight.view(-1)
        else:
            return (loss * weight.view(-1)).mean()


class ContinuousTimeConsistencyLoss(nn.Module):
    """Loss for continuous-time consistency training.

    Combines consistency matching with tangent matching for
    continuous-time formulation.
    """

    def __init__(
        self,
        tangent_weight: float = 1.0,
        consistency_weight: float = 1.0,
        base_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.tangent_weight = tangent_weight
        self.consistency_weight = consistency_weight
        self.base_loss = base_loss or PseudoHuberLoss()
        self.continuous_time = True  # Flag for trainer

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        student_tangent: Optional[torch.Tensor] = None,
        teacher_tangent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute continuous-time consistency loss.

        Args:
            student_pred: Student model prediction
            teacher_pred: Teacher model prediction
            student_tangent: Student tangent vector (dx/dt)
            teacher_tangent: Teacher tangent vector (dx/dt)

        Returns:
            Combined loss value
        """
        # Consistency loss
        consistency_loss = self.base_loss(student_pred, teacher_pred)

        # Tangent matching loss (if provided)
        if student_tangent is not None and teacher_tangent is not None:
            tangent_loss = F.mse_loss(student_tangent, teacher_tangent)
            total_loss = (
                self.consistency_weight * consistency_loss +
                self.tangent_weight * tangent_loss
            )
        else:
            total_loss = consistency_loss

        return total_loss


class DifferentialConsistencyLoss(nn.Module):
    """Differential consistency loss using finite differences.

    Approximates continuous-time consistency without explicit tangent computation.
    """

    def __init__(
        self,
        dt: float = 0.001,
        order: int = 1,
        base_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.dt = dt
        self.order = order
        self.base_loss = base_loss or PseudoHuberLoss()

    def forward(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        target_model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute differential consistency loss.

        Args:
            model: Student model
            x: Input at time t
            t: Time values
            target_model: Teacher model (if None, uses model)

        Returns:
            Loss value
        """
        if target_model is None:
            target_model = model

        # Current predictions
        pred_t = model(x, t)

        with torch.no_grad():
            target_pred_t = target_model(x, t)

        # Finite difference approximation
        if self.order == 1:
            # First-order approximation
            t_next = t + self.dt

            # Approximate next step
            sigma_t = model._t_to_sigma(t)
            sigma_next = model._t_to_sigma(t_next)

            # Simple Euler step
            eps = (x - pred_t) / sigma_t.view(-1, 1, 1, 1)
            x_next = pred_t + (sigma_next - sigma_t).view(-1, 1, 1, 1) * eps

            # Predictions at next step
            pred_next = model(x_next, t_next)

            with torch.no_grad():
                target_pred_next = target_model(x_next, t_next)

            # Finite difference
            d_student = (pred_next - pred_t) / self.dt
            d_teacher = (target_pred_next - target_pred_t) / self.dt

            # Loss
            loss = self.base_loss(d_student, d_teacher)

        else:
            # Higher-order approximations can be added
            raise NotImplementedError(f"Order {self.order} not implemented")

        return loss


class MultiScaleConsistencyLoss(nn.Module):
    """Multi-scale consistency loss for improved training.

    Computes consistency at multiple resolution scales.
    """

    def __init__(
        self,
        scales: list = [1, 2, 4],
        scale_weights: Optional[list] = None,
        base_loss: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.scales = scales
        self.scale_weights = scale_weights or [1.0] * len(scales)
        self.base_loss = base_loss or PseudoHuberLoss()

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-scale consistency loss.

        Args:
            student_pred: Student predictions
            teacher_pred: Teacher predictions

        Returns:
            Weighted multi-scale loss
        """
        total_loss = 0.0

        for scale, weight in zip(self.scales, self.scale_weights):
            if scale == 1:
                # Original scale
                loss = self.base_loss(student_pred, teacher_pred)
            else:
                # Downsampled scale
                student_down = F.avg_pool2d(student_pred, scale)
                teacher_down = F.avg_pool2d(teacher_pred, scale)
                loss = self.base_loss(student_down, teacher_down)

            total_loss = total_loss + weight * loss

        return total_loss


class HybridConsistencyLoss(nn.Module):
    """Hybrid loss combining consistency with other objectives.

    Used for CTM-style training with denoising and adversarial losses.
    """

    def __init__(
        self,
        consistency_weight: float = 1.0,
        denoising_weight: float = 0.1,
        adversarial_weight: float = 0.0,
        perceptual_weight: float = 0.0,
    ):
        super().__init__()

        self.consistency_weight = consistency_weight
        self.denoising_weight = denoising_weight
        self.adversarial_weight = adversarial_weight
        self.perceptual_weight = perceptual_weight

        # Base losses
        self.consistency_loss = PseudoHuberLoss()
        self.denoising_loss = nn.MSELoss()

        # Optional: perceptual loss
        if perceptual_weight > 0:
            from lpips import LPIPS
            self.perceptual_loss = LPIPS(net='vgg')
        else:
            self.perceptual_loss = None

    def forward(
        self,
        student_pred: torch.Tensor,
        teacher_pred: torch.Tensor,
        clean_images: Optional[torch.Tensor] = None,
        discriminator_real: Optional[torch.Tensor] = None,
        discriminator_fake: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute hybrid loss.

        Args:
            student_pred: Student predictions
            teacher_pred: Teacher predictions
            clean_images: Clean images for denoising loss
            discriminator_real: Discriminator output for real images
            discriminator_fake: Discriminator output for fake images

        Returns:
            Dictionary with individual losses and total loss
        """
        losses = {}

        # Consistency loss
        losses['consistency'] = self.consistency_loss(student_pred, teacher_pred)

        # Denoising loss
        if clean_images is not None and self.denoising_weight > 0:
            losses['denoising'] = self.denoising_loss(student_pred, clean_images)
        else:
            losses['denoising'] = torch.tensor(0.0, device=student_pred.device)

        # Adversarial loss
        if discriminator_fake is not None and self.adversarial_weight > 0:
            losses['adversarial'] = -torch.mean(discriminator_fake)
        else:
            losses['adversarial'] = torch.tensor(0.0, device=student_pred.device)

        # Perceptual loss
        if self.perceptual_loss is not None and clean_images is not None:
            losses['perceptual'] = self.perceptual_loss(
                student_pred, clean_images
            ).mean()
        else:
            losses['perceptual'] = torch.tensor(0.0, device=student_pred.device)

        # Total loss
        losses['total'] = (
            self.consistency_weight * losses['consistency'] +
            self.denoising_weight * losses['denoising'] +
            self.adversarial_weight * losses['adversarial'] +
            self.perceptual_weight * losses['perceptual']
        )

        return losses