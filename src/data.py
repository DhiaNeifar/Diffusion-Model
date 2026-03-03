from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

class ToMinusOneToOne:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * 2.0 - 1.0

def make_datasets(data_root: str = "./data", val_fraction: float = 0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        ToMinusOneToOne(),
    ])
    full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    n_val = int(len(full) * val_fraction)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    return train_ds, val_ds

def make_loaders(train_ds, val_ds, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
    )
    return train_loader, val_loader
