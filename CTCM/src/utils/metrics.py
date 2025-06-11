"""Evaluation metrics for consistency models.

Implements FID, Inception Score, Precision/Recall, and other metrics.
"""

import os
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from tqdm import tqdm
import torchvision.models as models
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3


class MetricsCalculator:
    """Calculate various metrics for generated images."""

    def __init__(self, device: torch.device = torch.device('cuda')):
        """Initialize metrics calculator.

        Args:
            device: Device to run calculations on
        """
        self.device = device

        # Load Inception model for FID/IS
        self.inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
        self.inception.eval()

        # Cache for reference statistics
        self.ref_stats_cache = {}

    @torch.no_grad()
    def calculate_fid(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        batch_size: int = 64,
    ) -> float:
        """Calculate Fréchet Inception Distance.

        Args:
            real_images: Real images tensor
            fake_images: Generated images tensor
            batch_size: Batch size for feature extraction

        Returns:
            FID score
        """
        # Extract features
        real_features = self._extract_features(real_images, batch_size)
        fake_features = self._extract_features(fake_images, batch_size)

        # Calculate statistics
        mu_real = np.mean(real_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)

        mu_fake = np.mean(fake_features, axis=0)
        sigma_fake = np.cov(fake_features, rowvar=False)

        # Calculate FID
        fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

        return fid

    @torch.no_grad()
    def calculate_inception_score(
        self,
        images: torch.Tensor,
        batch_size: int = 64,
        splits: int = 10,
    ) -> Tuple[float, float]:
        """Calculate Inception Score.

        Args:
            images: Generated images
            batch_size: Batch size for evaluation
            splits: Number of splits for IS calculation

        Returns:
            Tuple of (IS mean, IS std)
        """
        # Get predictions
        preds = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)

            # Resize if needed (Inception expects 299x299)
            if batch.shape[-1] != 299:
                batch = torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )

            # Get logits
            logits = self.inception(batch)[0]
            preds.append(torch.nn.functional.softmax(logits, dim=1).cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # Calculate IS
        scores = []
        for i in range(splits):
            part = preds[i::splits]
            kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))

        return float(np.mean(scores)), float(np.std(scores))

    @torch.no_grad()
    def calculate_precision_recall(
        self,
        real_features: np.ndarray,
        fake_features: np.ndarray,
        k: int = 3,
    ) -> Tuple[float, float]:
        """Calculate Precision and Recall metrics.

        Args:
            real_features: Features from real images
            fake_features: Features from generated images
            k: Number of nearest neighbors

        Returns:
            Tuple of (precision, recall)
        """
        # Compute pairwise distances
        real_distances = self._compute_pairwise_distances(real_features, real_features)
        fake_distances = self._compute_pairwise_distances(fake_features, fake_features)
        cross_distances = self._compute_pairwise_distances(fake_features, real_features)

        # Get k-NN radii
        real_radii = np.sort(real_distances, axis=1)[:, k]
        fake_radii = np.sort(fake_distances, axis=1)[:, k]

        # Precision: how many generated samples are near real samples
        precision = np.mean(np.min(cross_distances, axis=1) <= real_radii)

        # Recall: how many real samples are near generated samples
        recall = np.mean(np.min(cross_distances, axis=0) <= fake_radii)

        return float(precision), float(recall)

    def _extract_features(
        self,
        images: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """Extract Inception features from images.

        Args:
            images: Images tensor
            batch_size: Batch size for processing

        Returns:
            Features array
        """
        features = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(self.device)

            # Resize if needed
            if batch.shape[-1] != 299:
                batch = torch.nn.functional.interpolate(
                    batch, size=(299, 299), mode='bilinear', align_corners=False
                )

            # Extract features
            feat = self.inception(batch)[0]

            # Pool features
            if feat.dim() == 4:
                feat = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1))
                feat = feat.squeeze(-1).squeeze(-1)

            features.append(feat.cpu().numpy())

        return np.concatenate(features, axis=0)

    def _compute_pairwise_distances(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise distances between two sets of features."""
        dists = np.sum(X**2, axis=1, keepdims=True) + np.sum(Y**2, axis=1) - 2 * X @ Y.T
        return np.maximum(dists, 0.0)

    def get_reference_statistics(
        self,
        dataset_name: str,
        data_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get or compute reference statistics for a dataset.

        Args:
            dataset_name: Name of dataset
            data_dir: Directory containing dataset

        Returns:
            Tuple of (mean, covariance) for reference features
        """
        cache_key = f"{dataset_name}_{data_dir}"

        if cache_key in self.ref_stats_cache:
            return self.ref_stats_cache[cache_key]

        # Compute statistics
        if dataset_name == 'cifar10':
            from ..data.datasets import CIFAR10Dataset
            dataset = CIFAR10Dataset(root=data_dir, train=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Extract features
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=False, num_workers=4
        )

        features = []
        for batch in tqdm(dataloader, desc="Extracting reference features"):
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch

            feat = self._extract_features(images, batch_size=64)
            features.append(feat)

        features = np.concatenate(features, axis=0)

        # Compute statistics
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        # Cache
        self.ref_stats_cache[cache_key] = (mu, sigma)

        return mu, sigma


def compute_fid(
    model: nn.Module,
    reference_loader: torch.utils.data.DataLoader,
    num_samples: int = 50000,
    batch_size: int = 64,
    device: torch.device = torch.device('cuda'),
    num_steps: int = 1,
) -> float:
    """Compute FID score for a model.

    Args:
        model: Consistency model
        reference_loader: DataLoader for reference images
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        num_steps: Number of sampling steps

    Returns:
        FID score
    """
    calculator = MetricsCalculator(device)
    model.eval()

    # Generate samples
    generated = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            samples = model.sample(
                (batch_size, 3, 32, 32),
                num_steps=num_steps,
                device=device,
            )
            generated.append(samples.cpu())

    generated = torch.cat(generated, dim=0)[:num_samples]

    # Get reference images
    reference = []
    for batch in reference_loader:
        if isinstance(batch, (list, tuple)):
            images = batch[0]
        else:
            images = batch
        reference.append(images)

        if len(reference) * images.shape[0] >= num_samples:
            break

    reference = torch.cat(reference, dim=0)[:num_samples]

    # Calculate FID
    fid = calculator.calculate_fid(reference, generated, batch_size)

    return fid


def compute_inception_score(
    model: nn.Module,
    num_samples: int = 50000,
    batch_size: int = 64,
    device: torch.device = torch.device('cuda'),
    num_steps: int = 1,
    splits: int = 10,
) -> Tuple[float, float]:
    """Compute Inception Score for a model.

    Args:
        model: Consistency model
        num_samples: Number of samples to generate
        batch_size: Batch size for generation
        device: Device to use
        num_steps: Number of sampling steps
        splits: Number of splits for IS

    Returns:
        Tuple of (IS mean, IS std)
    """
    calculator = MetricsCalculator(device)
    model.eval()

    # Generate samples
    generated = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    with torch.no_grad():
        for _ in tqdm(range(num_batches), desc="Generating samples"):
            samples = model.sample(
                (batch_size, 3, 32, 32),
                num_steps=num_steps,
                device=device,
            )
            generated.append(samples.cpu())

    generated = torch.cat(generated, dim=0)[:num_samples]

    # Calculate IS
    is_mean, is_std = calculator.calculate_inception_score(generated, batch_size, splits)

    return is_mean, is_std