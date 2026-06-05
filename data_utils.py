from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transform(augment: str, train: bool) -> transforms.Compose:
    augment = augment.lower()
    if augment not in {"none", "rotate"}:
        raise ValueError("augment must be 'none' or 'rotate'")
    ops: list = []
    if train and augment == "rotate":
        ops.extend(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(12),
            ]
        )
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD))
    return transforms.Compose(ops)


def _maybe_limit(dataset, limit: int | None):
    if not limit or limit <= 0 or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def build_loaders(
    data_dir: str | Path,
    augment: str,
    batch_size: int,
    num_workers: int,
    download: bool = True,
    limit_train: int | None = None,
    limit_test: int | None = None,
):
    data_dir = Path(data_dir)
    train_set = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=build_transform(augment, train=True),
    )
    test_set = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=build_transform("none", train=False),
    )
    train_set = _maybe_limit(train_set, limit_train)
    test_set = _maybe_limit(test_set, limit_test)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader
