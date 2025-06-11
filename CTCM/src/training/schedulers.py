"""Schedulers for progressive training and noise schedules."""

import math
from typing import Optional, Union

import torch
import numpy as np


class ProgressiveSchedule:
    """Progressive training schedule for consistency models.

    Gradually increases the number of discretization steps during training.
    """

    def __init__(
        self,
        initial_N: int = 4,
        final_N: int = 1280,
        total_iterations: int = 100000,
        doubling_iterations: int = 20000,
        schedule_type: str = 'geometric',
    ):
        """Initialize progressive schedule.

        Args:
            initial_N: Starting discretization level
            final_N: Final discretization level
            total_iterations: Total training iterations
            doubling_iterations: Iterations between doublings
            schedule_type: Type of schedule ('geometric' or 'linear')
        """
        self.initial_N = initial_N
        self.final_N = final_N
        self.total_iterations = total_iterations
        self.doubling_iterations = doubling_iterations
        self.schedule_type = schedule_type

        # Compute schedule milestones
        self._compute_milestones()

    def _compute_milestones(self):
        """Compute iteration milestones for N changes."""
        if self.schedule_type == 'geometric':
            # Geometric progression (doubling)
            self.milestones = []
            N = self.initial_N
            iteration = 0

            while N < self.final_N and iteration < self.total_iterations:
                self.milestones.append((iteration, N))
                iteration += self.doubling_iterations
                N = min(N * 2, self.final_N)

            # Add final milestone
            self.milestones.append((iteration, self.final_N))

        elif self.schedule_type == 'linear':
            # Linear progression
            num_steps = 10
            iterations = np.linspace(0, self.total_iterations, num_steps + 1)
            Ns = np.linspace(self.initial_N, self.final_N, num_steps + 1)

            self.milestones = [
                (int(it), int(N)) for it, N in zip(iterations, Ns)
            ]

    def get_N(self, iteration: int) -> int:
        """Get discretization level for current iteration.

        Args:
            iteration: Current training iteration

        Returns:
            Current discretization level N
        """
        # Find appropriate milestone
        for i in range(len(self.milestones) - 1):
            if iteration < self.milestones[i + 1][0]:
                return self.milestones[i][1]

        # Return final N
        return self.milestones[-1][1]

    def get_progress(self, iteration: int) -> float:
        """Get training progress as fraction.

        Args:
            iteration: Current iteration

        Returns:
            Progress in [0, 1]
        """
        return min(iteration / self.total_iterations, 1.0)


class KarrasSchedule:
    """Karras noise schedule for diffusion models."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        rho: float = 7.0,
    ):
        """Initialize Karras schedule.

        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            rho: Schedule exponent
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho

    def get_sigmas(
        self,
        n: int,
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
        """Get noise schedule with n steps.

        Args:
            n: Number of discretization steps
            device: Device for tensor

        Returns:
            Noise levels of shape [n]
        """
        # Karras schedule
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / self.rho)
        max_inv_rho = self.sigma_max ** (1 / self.rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** self.rho

        return sigmas

    def get_truncated_sigmas(
        self,
        n: int,
        truncation: float = 0.1,
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
        """Get truncated noise schedule.

        Useful for training on cleaner images.

        Args:
            n: Number of steps
            truncation: Truncation factor (0 = full truncation, 1 = no truncation)
            device: Device for tensor

        Returns:
            Truncated noise levels
        """
        # Adjust sigma_max based on truncation
        sigma_max_truncated = self.sigma_min + truncation * (self.sigma_max - self.sigma_min)

        # Create truncated schedule
        schedule = KarrasSchedule(self.sigma_min, sigma_max_truncated, self.rho)
        return schedule.get_sigmas(n, device)


class CosineSchedule:
    """Cosine noise schedule."""

    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        s: float = 0.008,
    ):
        """Initialize cosine schedule.

        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            s: Small offset to prevent singularity
        """
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.s = s

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Compute alpha_bar for time t in [0, 1]."""
        return torch.cos((t + self.s) / (1 + self.s) * math.pi / 2) ** 2

    def get_sigmas(
        self,
        n: int,
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
        """Get noise schedule with n steps."""
        t = torch.linspace(0, 1, n, device=device)
        alpha_bars = self.alpha_bar(t)

        # Convert to sigmas
        sigmas = torch.sqrt((1 - alpha_bars) / alpha_bars)

        # Scale to desired range
        sigmas = sigmas * (self.sigma_max - self.sigma_min) + self.sigma_min

        return sigmas


class AdaptiveSchedule:
    """Adaptive schedule that adjusts based on training progress."""

    def __init__(
        self,
        base_schedule: Union[KarrasSchedule, CosineSchedule],
        warmup_iterations: int = 10000,
        adapt_iterations: int = 50000,
    ):
        """Initialize adaptive schedule.

        Args:
            base_schedule: Base noise schedule
            warmup_iterations: Warmup period
            adapt_iterations: Adaptation period
        """
        self.base_schedule = base_schedule
        self.warmup_iterations = warmup_iterations
        self.adapt_iterations = adapt_iterations

        # Adaptation parameters
        self.truncation_start = 1.0
        self.truncation_end = 0.5

    def get_sigmas(
        self,
        n: int,
        iteration: int,
        device: torch.device = torch.device('cuda'),
    ) -> torch.Tensor:
        """Get adaptive noise schedule.

        Args:
            n: Number of steps
            iteration: Current training iteration
            device: Device for tensor

        Returns:
            Adaptive noise levels
        """
        if iteration < self.warmup_iterations:
            # Full schedule during warmup
            return self.base_schedule.get_sigmas(n, device)

        elif iteration < self.warmup_iterations + self.adapt_iterations:
            # Gradually truncate schedule
            progress = (iteration - self.warmup_iterations) / self.adapt_iterations
            truncation = self.truncation_start + progress * (
                self.truncation_end - self.truncation_start
            )

            if hasattr(self.base_schedule, 'get_truncated_sigmas'):
                return self.base_schedule.get_truncated_sigmas(n, truncation, device)
            else:
                # Fallback: simple truncation
                sigmas = self.base_schedule.get_sigmas(n, device)
                max_idx = int(n * truncation)
                return sigmas[:max_idx]

        else:
            # Final truncated schedule
            if hasattr(self.base_schedule, 'get_truncated_sigmas'):
                return self.base_schedule.get_truncated_sigmas(
                    n, self.truncation_end, device
                )
            else:
                sigmas = self.base_schedule.get_sigmas(n, device)
                max_idx = int(n * self.truncation_end)
                return sigmas[:max_idx]


class LearningRateSchedule:
    """Custom learning rate schedules for consistency training."""

    @staticmethod
    def get_polynomial_schedule(
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        lr_end: float = 1e-7,
        power: float = 1.0,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create polynomial decay schedule."""

        def lr_lambda(current_step: int):
            if current_step > num_training_steps:
                return lr_end / optimizer.param_groups[0]['lr']
            else:
                lr_range = optimizer.param_groups[0]['lr'] - lr_end
                decay = (1 - current_step / num_training_steps) ** power
                return (lr_end + lr_range * decay) / optimizer.param_groups[0]['lr']

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch
        )

    @staticmethod
    def get_warmup_cosine_schedule(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> torch.optim.lr_scheduler.LambdaLR:
        """Create warmup + cosine decay schedule."""

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            progress = float(current_step - num_warmup_steps) / float(
                max(1, num_training_steps - num_warmup_steps)
            )
            return max(
                0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch
        )