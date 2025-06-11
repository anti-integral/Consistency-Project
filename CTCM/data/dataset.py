import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def get_cifar_dataloader(batch_size: int, img_size: int, num_workers: int = 4):
    """
    Returns a CIFAR-10 training dataloader with simple augmentations.
    """
    transform = T.Compose([
        T.Resize(img_size),
        T.RandomHorizontalFlip(),
        T.ToTensor(),                      # [0,1]
        T.Normalize((0.5, 0.5, 0.5),       # center to [-1,1]
                    (0.5, 0.5, 0.5)),
    ])
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
