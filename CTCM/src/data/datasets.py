"""Dataset utilities for consistency model training.

Supports CIFAR-10, ImageNet, and custom datasets.
"""

import os
from typing import Optional, Callable, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class CIFAR10Dataset(Dataset):
    """CIFAR-10 dataset with augmentations for consistency training."""

    def __init__(
        self,
        root: str = './data',
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = True,
        use_labels: bool = False,
    ):
        """Initialize CIFAR-10 dataset.

        Args:
            root: Root directory for data
            train: Whether to use training set
            transform: Optional transform
            download: Whether to download if not found
            use_labels: Whether to return labels
        """
        self.use_labels = use_labels

        # Default transform
        if transform is None:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        # Load CIFAR-10
        self.dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        img, label = self.dataset[idx]

        if self.use_labels:
            return img, label
        else:
            return img


class ImageNetDataset(Dataset):
    """ImageNet dataset for consistency model training."""

    def __init__(
        self,
        root: str,
        split: str = 'train',
        image_size: int = 64,
        transform: Optional[Callable] = None,
        use_labels: bool = True,
    ):
        """Initialize ImageNet dataset.

        Args:
            root: Root directory containing ImageNet
            split: 'train' or 'val'
            image_size: Size to resize images to
            transform: Optional transform
            use_labels: Whether to return labels
        """
        self.root = os.path.join(root, split)
        self.image_size = image_size
        self.use_labels = use_labels

        # Default transform
        if transform is None:
            if split == 'train':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(int(image_size * 1.1)),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.transform = transform

        # Load dataset
        self.dataset = torchvision.datasets.ImageFolder(self.root)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        path, label = self.dataset.samples[idx]

        # Load image
        img = Image.open(path).convert('RGB')

        # Apply transform
        if self.transform is not None:
            img = self.transform(img)

        if self.use_labels:
            return img, label
        else:
            return img


class LSUNDataset(Dataset):
    """LSUN dataset for consistency model training."""

    def __init__(
        self,
        root: str,
        category: str = 'bedroom',
        split: str = 'train',
        image_size: int = 256,
        transform: Optional[Callable] = None,
    ):
        """Initialize LSUN dataset.

        Args:
            root: Root directory
            category: LSUN category
            split: 'train' or 'val'
            image_size: Size to resize images to
            transform: Optional transform
        """
        from torchvision.datasets import LSUN

        # Default transform
        if transform is None:
            if split == 'train':
                transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])
            else:
                transform = transforms.Compose([
                    transforms.Resize(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        # Load dataset
        self.dataset = LSUN(
            root=root,
            classes=[f'{category}_{split}'],
            transform=transform,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.dataset[idx]
        return img


class PrecomputedDataset(Dataset):
    """Dataset for pre-computed trajectories (for DSNO-style training)."""

    def __init__(
        self,
        data_path: str,
        num_timesteps: int = 10,
        return_trajectory: bool = False,
    ):
        """Initialize pre-computed dataset.

        Args:
            data_path: Path to pre-computed data
            num_timesteps: Number of timesteps in trajectory
            return_trajectory: Whether to return full trajectory
        """
        self.data_path = data_path
        self.num_timesteps = num_timesteps
        self.return_trajectory = return_trajectory

        # Load data index
        self.data_files = sorted([
            f for f in os.listdir(data_path)
            if f.endswith('.npz') or f.endswith('.pt')
        ])

    def __len__(self) -> int:
        return len(self.data_files)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        file_path = os.path.join(self.data_path, self.data_files[idx])

        # Load data
        if file_path.endswith('.npz'):
            data = np.load(file_path)
            x_0 = torch.from_numpy(data['x_0']).float()
            trajectory = torch.from_numpy(data['trajectory']).float()
            timesteps = torch.from_numpy(data['timesteps']).float()
        else:
            data = torch.load(file_path)
            x_0 = data['x_0']
            trajectory = data['trajectory']
            timesteps = data['timesteps']

        if self.return_trajectory:
            return x_0, trajectory, timesteps
        else:
            # Return random point on trajectory
            t_idx = torch.randint(0, len(timesteps), (1,)).item()
            return trajectory[t_idx], timesteps[t_idx]


def get_dataloader(
    config: dict,
    split: str = 'train',
    shuffle: Optional[bool] = None,
    drop_last: Optional[bool] = None,
) -> DataLoader:
    """Create dataloader from config.

    Args:
        config: Data configuration
        split: 'train' or 'val'
        shuffle: Whether to shuffle (defaults based on split)
        drop_last: Whether to drop last batch

    Returns:
        DataLoader instance
    """
    dataset_name = config['data']['dataset']
    batch_size = config['training']['batch_size']

    # Default shuffle/drop settings
    if shuffle is None:
        shuffle = (split == 'train')
    if drop_last is None:
        drop_last = (split == 'train')

    # Create dataset
    if dataset_name == 'cifar10':
        dataset = CIFAR10Dataset(
            root=config['paths']['data_dir'],
            train=(split == 'train'),
            use_labels=config['data'].get('use_labels', False),
        )
    elif dataset_name == 'imagenet':
        dataset = ImageNetDataset(
            root=config['paths']['data_dir'],
            split=split,
            image_size=config['data']['image_size'],
            use_labels=config['data'].get('use_labels', True),
        )
    elif dataset_name == 'lsun':
        dataset = LSUNDataset(
            root=config['paths']['data_dir'],
            category=config['data'].get('category', 'bedroom'),
            split=split,
            image_size=config['data']['image_size'],
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True,
    )

    return dataloader


class InfiniteDataLoader:
    """Infinite dataloader that cycles through dataset."""

    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            return next(self.iterator)