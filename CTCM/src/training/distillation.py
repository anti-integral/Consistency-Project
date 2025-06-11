"""Distillation utilities for training consistency models from diffusion models.

Implements ECT-style efficient distillation and trajectory sampling.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from tqdm import tqdm

from ..models.consistency_model import ConsistencyModel


class DiffusionDistiller:
    """Handles distillation from pre-trained diffusion models."""

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: ConsistencyModel,
        config: Dict,
    ):
        """Initialize distiller.

        Args:
            teacher_model: Pre-trained diffusion model
            student_model: Consistency model to train
            config: Distillation configuration
        """
        self.teacher = teacher_model
        self.student = student_model
        self.config = config

        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

        # ODE solver settings
        self.solver_type = config.get('solver', 'euler')
        self.solver_steps = config.get('solver_steps', 10)

    def sample_trajectory(
        self,
        x_0: torch.Tensor,
        num_steps: int = 50,
        return_all: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Sample trajectory from teacher model.

        Args:
            x_0: Clean images
            num_steps: Number of ODE steps
            return_all: Whether to return all intermediate states

        Returns:
            Final sample or list of all samples
        """
        # Get noise schedule
        sigmas = self.student.get_karras_sigmas(num_steps + 1)

        # Add initial noise
        x_t = x_0 + sigmas[0] * torch.randn_like(x_0)

        # Store trajectory if needed
        if return_all:
            trajectory = [x_t]

        # Solve ODE
        for i in range(num_steps):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]
            t_curr = self.student._sigma_to_t(sigma_curr)

            # Get teacher prediction
            with torch.no_grad():
                if hasattr(self.teacher, 'forward_diffusion'):
                    # Handle different model interfaces
                    v_pred = self.teacher.forward_diffusion(x_t, t_curr)
                else:
                    v_pred = self.teacher(x_t, t_curr)

            # ODE step
            if self.solver_type == 'euler':
                x_t = self._euler_step(x_t, v_pred, sigma_curr, sigma_next)
            elif self.solver_type == 'heun':
                x_t = self._heun_step(x_t, t_curr, sigma_curr, sigma_next)
            else:
                raise ValueError(f"Unknown solver: {self.solver_type}")

            if return_all:
                trajectory.append(x_t)

        if return_all:
            return trajectory
        else:
            return x_t

    def _euler_step(
        self,
        x: torch.Tensor,
        v: torch.Tensor,
        sigma_curr: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Euler ODE step."""
        dt = sigma_next - sigma_curr
        return x + dt * v

    def _heun_step(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        sigma_curr: torch.Tensor,
        sigma_next: torch.Tensor,
    ) -> torch.Tensor:
        """Heun's method (improved Euler)."""
        # First step
        with torch.no_grad():
            v1 = self.teacher(x, t)
        x_euler = x + (sigma_next - sigma_curr) * v1

        # Second step
        t_next = self.student._sigma_to_t(sigma_next)
        with torch.no_grad():
            v2 = self.teacher(x_euler, t_next)

        # Average
        x_next = x + (sigma_next - sigma_curr) * (v1 + v2) / 2

        return x_next

    def generate_training_pairs(
        self,
        x_0: torch.Tensor,
        num_pairs: int = 2,
        min_sigma: Optional[float] = None,
        max_sigma: Optional[float] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Generate training pairs for consistency distillation.

        Args:
            x_0: Clean images
            num_pairs: Number of pairs per image
            min_sigma: Minimum noise level
            max_sigma: Maximum noise level

        Returns:
            List of (x_t, x_s, t) tuples for training
        """
        if min_sigma is None:
            min_sigma = self.student.sigma_min
        if max_sigma is None:
            max_sigma = self.student.sigma_max

        pairs = []

        for _ in range(num_pairs):
            # Sample random noise levels
            log_sigma = torch.rand(x_0.shape[0], device=x_0.device) * (
                torch.log(torch.tensor(max_sigma)) -
                torch.log(torch.tensor(min_sigma))
            ) + torch.log(torch.tensor(min_sigma))
            sigma_t = torch.exp(log_sigma)

            # Add noise
            noise = torch.randn_like(x_0)
            x_t = x_0 + sigma_t.view(-1, 1, 1, 1) * noise

            # Get teacher prediction
            t = self.student._sigma_to_t(sigma_t)
            with torch.no_grad():
                x_s = self.teacher(x_t, t)

            pairs.append((x_t, x_s, t))

        return pairs


class ECTDistiller(DiffusionDistiller):
    """Easy Consistency Tuning distiller.

    Implements the efficient distillation approach from ECT paper.
    """

    def __init__(
        self,
        teacher_model: nn.Module,
        student_model: ConsistencyModel,
        config: Dict,
    ):
        super().__init__(teacher_model, student_model, config)

        # ECT-specific settings
        self.use_teacher_init = config.get('use_teacher_init', True)
        self.progressive_matching = config.get('progressive_matching', True)

        # Initialize student from teacher if requested
        if self.use_teacher_init:
            self._initialize_from_teacher()

    def _initialize_from_teacher(self):
        """Initialize student weights from teacher."""
        # Copy matching parameters
        teacher_state = self.teacher.state_dict()
        student_state = self.student.state_dict()

        for key in student_state:
            if key in teacher_state and student_state[key].shape == teacher_state[key].shape:
                student_state[key] = teacher_state[key].clone()

        self.student.load_state_dict(student_state)

    def compute_ect_loss(
        self,
        x_0: torch.Tensor,
        loss_fn: nn.Module,
        num_steps: int = 18,
    ) -> torch.Tensor:
        """Compute ECT loss for batch.

        Args:
            x_0: Clean images
            loss_fn: Loss function to use
            num_steps: Number of discretization steps

        Returns:
            Loss value
        """
        batch_size = x_0.shape[0]
        device = x_0.device

        # Get noise schedule
        sigmas = self.student.get_karras_sigmas(num_steps, device)

        # Sample time indices
        indices = torch.randint(0, num_steps - 1, (batch_size,), device=device)

        # Get noise levels
        sigma_curr = sigmas[indices]
        sigma_next = sigmas[indices + 1]

        # Convert to time
        t_curr = self.student._sigma_to_t(sigma_curr)
        t_next = self.student._sigma_to_t(sigma_next)

        # Add noise
        noise = torch.randn_like(x_0)
        x_curr = x_0 + sigma_curr.view(-1, 1, 1, 1) * noise

        # Teacher trajectory step
        with torch.no_grad():
            # Get velocity from teacher
            v_teacher = self.teacher(x_curr, t_curr)

            # Compute next point on trajectory
            if self.progressive_matching and num_steps > 18:
                # Use higher-order solver for better targets
                x_next = self._heun_step(x_curr, t_curr, sigma_curr, sigma_next)
            else:
                # Simple Euler step
                x_next = x_curr + (sigma_next - sigma_curr).view(-1, 1, 1, 1) * v_teacher

        # Student predictions
        student_curr = self.student(x_curr, t_curr)
        student_next = self.student(x_next, t_next)

        # Consistency loss
        loss = loss_fn(student_curr, student_next.detach())

        return loss

    def compute_trajectory_matching_loss(
        self,
        x_0: torch.Tensor,
        loss_fn: nn.Module,
        trajectory_steps: List[int] = [10, 50],
    ) -> torch.Tensor:
        """Compute trajectory matching loss.

        Matches student predictions along entire trajectories.

        Args:
            x_0: Clean images
            loss_fn: Loss function
            trajectory_steps: List of trajectory lengths to use

        Returns:
            Trajectory matching loss
        """
        total_loss = 0.0

        for steps in trajectory_steps:
            # Generate teacher trajectory
            trajectory = self.sample_trajectory(x_0, steps, return_all=True)

            # Compute student predictions along trajectory
            for i in range(len(trajectory) - 1):
                x_t = trajectory[i]

                # Get time
                sigma_t = self.student.get_karras_sigmas(steps + 1)[i]
                t = self.student._sigma_to_t(sigma_t)

                # Student prediction
                student_pred = self.student(x_t, t)

                # Target is the final point
                target = trajectory[-1]

                # Add to loss
                total_loss = total_loss + loss_fn(student_pred, target.detach())

        return total_loss / (len(trajectory_steps) * steps)