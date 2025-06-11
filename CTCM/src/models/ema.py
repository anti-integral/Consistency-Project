"""Exponential Moving Average for model parameters.

Used for stable training and evaluation.
"""

from typing import Optional, Union
import copy

import torch
import torch.nn as nn


class ExponentialMovingAverage:
    """Maintains exponential moving averages of model parameters.

    Implements the improved EMA from TrigFlow that avoids the theoretical
    flaw in the original consistency model formulation.
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        use_num_updates: bool = True,
        power: float = 2/3,
    ):
        """Initialize EMA.

        Args:
            model: Model to track
            decay: Base decay rate
            use_num_updates: Whether to ramp up decay rate
            power: Power for decay schedule
        """
        self.model = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.power = power
        self.num_updates = 0

        # Initialize shadow parameters
        self.shadow_params = [
            p.clone().detach() for p in model.parameters()
        ]

        # Store device
        self.device = next(model.parameters()).device

    def update(self, model: Optional[nn.Module] = None):
        """Update EMA parameters.

        Args:
            model: Model with new parameters (if None, uses self.model)
        """
        if model is None:
            model = self.model

        # Compute adaptive decay if enabled
        decay = self.decay
        if self.use_num_updates:
            self.num_updates += 1
            # Ramp up decay rate over time
            decay = min(
                self.decay,
                (1 + self.num_updates) / (10 + self.num_updates)
            )

        # Update shadow parameters
        with torch.no_grad():
            for shadow, param in zip(
                self.shadow_params, model.parameters()
            ):
                shadow.sub_((1.0 - decay) * (shadow - param))

    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for shadow, param in zip(
            self.shadow_params, self.model.parameters()
        ):
            param.data.copy_(shadow.data)

    def restore(self):
        """Restore original parameters."""
        for shadow, param in zip(
            self.shadow_params, self.model.parameters()
        ):
            shadow.data.copy_(param.data)

    def state_dict(self):
        """Get state dict for saving."""
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow_params': self.shadow_params,
        }

    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']

    def to(self, device: Union[str, torch.device]):
        """Move EMA to device."""
        self.shadow_params = [
            p.to(device) for p in self.shadow_params
        ]
        self.device = device
        return self

    def copy_to(self, model: nn.Module):
        """Copy EMA parameters to another model.

        Args:
            model: Target model
        """
        with torch.no_grad():
            for shadow, param in zip(
                self.shadow_params, model.parameters()
            ):
                param.data.copy_(shadow.data)

    @torch.no_grad()
    def ema_model(self, model: Optional[nn.Module] = None) -> nn.Module:
        """Get a copy of the model with EMA parameters.

        Args:
            model: Base model (if None, uses self.model)

        Returns:
            Model with EMA parameters
        """
        if model is None:
            model = self.model

        # Create a deep copy of the model
        ema_model = copy.deepcopy(model)
        
        # Apply EMA parameters
        for shadow, param in zip(
            self.shadow_params, ema_model.parameters()
        ):
            param.data.copy_(shadow.data)

        return ema_model

    def update_with_teacher(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        decay: Optional[float] = None,
    ):
        """Update EMA using teacher-student setup.

        This is used in TrigFlow to avoid the theoretical flaw where
        the EMA model could reference itself.

        Args:
            student_model: Student model being trained
            teacher_model: Teacher model (typically previous checkpoint)
            decay: Optional custom decay rate
        """
        if decay is None:
            decay = self.decay

        with torch.no_grad():
            # Update from teacher, not from student
            for shadow, teacher_param, student_param in zip(
                self.shadow_params,
                teacher_model.parameters(),
                student_model.parameters(),
            ):
                # Weighted average between teacher and student
                shadow.mul_(decay).add_(
                    teacher_param.data, alpha=(1 - decay)
                )