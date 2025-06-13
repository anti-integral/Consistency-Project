import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, RandomSampler

_CIFAR_STATS = {
    "mean": (0.4914, 0.4822, 0.4465),
    "std":  (0.2470, 0.2435, 0.2616),
}

def get_cifar_loaders(batch_size: int, num_workers: int = 4):
    """Return (train_loader, test_loader) for CIFAR‑10."""
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_STATS["mean"], _CIFAR_STATS["std"]),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_STATS["mean"], _CIFAR_STATS["std"]),
    ])
    root = "./data"
    train_set = datasets.CIFAR10(root, train=True, transform=tf_train, download=True)
    test_set  = datasets.CIFAR10(root, train=False, transform=tf_test,  download=True)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=RandomSampler(train_set, replacement=True),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader
