"""Main trainer for Neural Operator Continuous Time Consistency Models."""

import os
from typing import Dict, Optional, Union
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from ..models.consistency_model import ConsistencyModel
from ..models.ema import ExponentialMovingAverage
from .losses import PseudoHuberLoss, ContinuousTimeConsistencyLoss
from .schedulers import ProgressiveSchedule, KarrasSchedule
from ..utils.metrics import compute_fid, compute_inception_score
from ..utils.helpers import save_checkpoint, load_checkpoint


logger = logging.getLogger(__name__)


class ConsistencyTrainer:
    """Trainer for consistency models with various training modes."""

    def __init__(
        self,
        model: ConsistencyModel,
        config: Dict,
        device: torch.device = torch.device("cuda"),
    ):
        """Initialize trainer.

        Args:
            model: Consistency model to train
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training mode
        self.mode = config['training']['mode']
        self.iterations = config['training']['total_iterations']

        # Initialize components
        self._setup_optimizer()
        self._setup_loss()
        self._setup_scheduler()
        self._setup_ema()

        # Setup logging
        self.use_wandb = config.get('wandb', {}).get('enabled', False) and WANDB_AVAILABLE
        if self.use_wandb:
            self._setup_wandb()
        elif config.get('wandb', {}).get('enabled', False):
            logger.warning("wandb requested but not installed. Install with: pip install wandb")

        # Mixed precision training
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = config['paths']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # State
        self.current_iteration = 0
        self.best_fid = float('inf')

    def _setup_optimizer(self):
        """Setup optimizer."""
        opt_config = self.config['training']['optimizer']

        if opt_config['type'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay'],
                eps=opt_config['eps'],
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_config['type']}")

    def _setup_loss(self):
        """Setup loss function."""
        loss_config = self.config['training']['loss']

        if loss_config['type'] == 'pseudo_huber':
            self.loss_fn = PseudoHuberLoss(
                c=loss_config['c'],
                reduction=loss_config['reduction'],
            )
        elif loss_config['type'] == 'continuous_time':
            self.loss_fn = ContinuousTimeConsistencyLoss(
                loss_config.get('tangent_weight', 1.0),
                loss_config.get('consistency_weight', 1.0),
            )
        else:
            self.loss_fn = nn.MSELoss()

    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        scheduler_config = self.config['training']['lr_scheduler']

        if scheduler_config['type'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.iterations,
                eta_min=scheduler_config['min_lr'],
            )
        elif scheduler_config['type'] == 'polynomial':
            self.scheduler = torch.optim.lr_scheduler.PolynomialLR(
                self.optimizer,
                total_iters=self.iterations,
                power=scheduler_config.get('power', 0.9),
            )
        else:
            self.scheduler = None

        # Progressive training schedule
        if self.config['training']['progressive']['enabled']:
            prog_config = self.config['training']['progressive']
            self.progressive_schedule = ProgressiveSchedule(
                initial_N=prog_config['initial_discretization'],
                final_N=prog_config['final_discretization'],
                total_iterations=self.iterations,
                doubling_iterations=prog_config['doubling_iterations'],
            )
        else:
            self.progressive_schedule = None

        # Karras noise schedule
        self.noise_schedule = KarrasSchedule(
            sigma_min=self.model.sigma_min,
            sigma_max=self.model.sigma_max,
            rho=self.model.rho,
        )

    def _setup_ema(self):
        """Setup exponential moving average."""
        ema_config = self.config['training']['ema']

        if ema_config['enabled']:
            self.ema = ExponentialMovingAverage(
                self.model,
                decay=ema_config['decay'],
                use_num_updates=True,
            )
        else:
            self.ema = None

    def _setup_wandb(self):
        """Setup Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            logger.warning("wandb not available")
            return

        wandb_config = self.config['wandb']

        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config.get('entity'),
            tags=wandb_config.get('tags', []),
            config=self.config,
        )

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        teacher_model: Optional[nn.Module] = None,
    ):
        """Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            teacher_model: Teacher model for distillation
        """
        logger.info(f"Starting training in {self.mode} mode")

        # Training loop
        progress_bar = tqdm(range(self.iterations), desc="Training")
        data_iter = iter(train_loader)

        for iteration in progress_bar:
            self.current_iteration = iteration

            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

            # Move to device
            if isinstance(batch, (list, tuple)):
                x, _ = batch
            else:
                x = batch
            x = x.to(self.device)

            # Training step
            if self.mode == 'distillation':
                loss = self._distillation_step(x, teacher_model)
            elif self.mode == 'consistency_training':
                loss = self._consistency_training_step(x)
            else:
                raise ValueError(f"Unknown training mode: {self.mode}")

            # Update progress bar
            progress_bar.set_postfix(loss=loss.item())

            # Logging
            if iteration % self.config['training']['log_every'] == 0:
                self._log_metrics({'loss': loss.item()}, iteration)

            # Validation
            if iteration % self.config['training']['sample_every'] == 0:
                self._validate(val_loader, iteration)

            # Checkpointing
            if iteration % self.config['training']['checkpoint']['save_every'] == 0:
                self._save_checkpoint(iteration)

        # Final checkpoint
        self._save_checkpoint(self.iterations, final=True)

    def _distillation_step(
        self,
        x: torch.Tensor,
        teacher_model: nn.Module,
    ) -> torch.Tensor:
        """Distillation training step.

        Args:
            x: Clean images
            teacher_model: Teacher diffusion model

        Returns:
            Loss value
        """
        batch_size = x.shape[0]

        # Get current discretization level
        if self.progressive_schedule is not None:
            N = self.progressive_schedule.get_N(self.current_iteration)
        else:
            N = 50

        # Sample time steps
        sigmas = self.noise_schedule.get_sigmas(N)
        indices = torch.randint(0, N - 1, (batch_size,), device=self.device)

        t_curr = self.model._sigma_to_t(sigmas[indices])
        t_next = self.model._sigma_to_t(sigmas[indices + 1])

        # Add noise
        noise = torch.randn_like(x)
        x_curr = x + sigmas[indices].view(-1, 1, 1, 1) * noise

        # Teacher prediction
        with torch.no_grad():
            # Get teacher trajectory
            teacher_pred = teacher_model(x_curr, t_curr)

            # Compute next step using ODE solver
            velocity = (x_curr - teacher_pred) / sigmas[indices].view(-1, 1, 1, 1)
            x_next = x_curr - (sigmas[indices] - sigmas[indices + 1]).view(-1, 1, 1, 1) * velocity

        # Student prediction
        with autocast(enabled=self.use_amp):
            student_pred_curr = self.model(x_curr, t_curr)
            student_pred_next = self.model(x_next, t_next)

            # Consistency loss
            loss = self.loss_fn(student_pred_curr, student_pred_next.detach())

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def _consistency_training_step(self, x: torch.Tensor) -> torch.Tensor:
        """Consistency training step (training from scratch).

        Args:
            x: Clean images

        Returns:
            Loss value
        """
        batch_size = x.shape[0]

        # Get current discretization level
        if self.progressive_schedule is not None:
            N = self.progressive_schedule.get_N(self.current_iteration)
        else:
            N = 18

        # Sample time steps
        sigmas = self.noise_schedule.get_sigmas(N)
        indices = torch.randint(0, N - 1, (batch_size,), device=self.device)

        t_curr = self.model._sigma_to_t(sigmas[indices])
        t_next = self.model._sigma_to_t(sigmas[indices + 1])

        # Add noise
        noise = torch.randn_like(x)
        x_curr = x + sigmas[indices].view(-1, 1, 1, 1) * noise
        x_next = x + sigmas[indices + 1].view(-1, 1, 1, 1) * noise

        # Forward pass
        with autocast(enabled=self.use_amp):
            # Current prediction
            pred_curr = self.model(x_curr, t_curr)

            # Next prediction (with stop gradient)
            with torch.no_grad():
                if self.ema is not None:
                    # Use EMA model for target
                    ema_model = self.ema.ema_model(self.model)
                    pred_next = ema_model(x_next, t_next)
                else:
                    pred_next = self.model(x_next, t_next)

            # Consistency loss
            loss = self.loss_fn(pred_curr, pred_next)

            # Optional: Add denoising score matching loss
            if self.config['training'].get('hybrid_training', {}).get('enabled', False):
                dsm_weight = self.config['training']['hybrid_training']['dsm_weight']
                if self.current_iteration >= self.config['training']['hybrid_training']['dsm_start_iter']:
                    # Denoising loss
                    dsm_loss = F.mse_loss(pred_curr, x)
                    loss = loss + dsm_weight * dsm_loss

        # Backward pass
        self.optimizer.zero_grad()

        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()

        # Update EMA
        if self.ema is not None:
            self.ema.update()

        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()

        return loss

    def _validate(self, val_loader: Optional[DataLoader], iteration: int):
        """Run validation and generate samples."""
        if val_loader is None:
            return

        self.model.eval()

        # Generate samples
        num_samples = 64
        samples = self.model.sample(
            (num_samples, 3, 32, 32),
            num_steps=self.config['sampling']['num_steps'],
            device=self.device,
        )

        # Log samples
        if self.use_wandb:
            import torchvision
            wandb.log({
                'samples': wandb.Image(
                    torchvision.utils.make_grid(
                        samples, nrow=8, normalize=True, value_range=(-1, 1)
                    )
                )
            }, step=iteration)

        self.model.train()

    def _log_metrics(self, metrics: Dict[str, float], iteration: int):
        """Log metrics."""
        # Add learning rate
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        # Add discretization level if using progressive schedule
        if self.progressive_schedule is not None:
            metrics['N'] = self.progressive_schedule.get_N(iteration)

        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=iteration)

        # Log to console
        if iteration % 1000 == 0:
            logger.info(
                f"Iteration {iteration}: " +
                ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            )

    def _save_checkpoint(self, iteration: int, final: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_fid': self.best_fid,
        }

        if self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save checkpoint
        if final:
            path = os.path.join(self.checkpoint_dir, 'final_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_{iteration}.pt')

        save_checkpoint(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")