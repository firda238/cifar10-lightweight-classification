from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, utils as tv_utils

from data_utils import CIFAR10_CLASSES
from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Save a CIFAR-10 sample grid.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="results/cifar10_samples.png")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--no-download", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = datasets.CIFAR10(
        root=args.data_dir,
        train=True,
        download=not args.no_download,
        transform=transforms.ToTensor(),
    )
    images = []
    labels = []
    for idx in range(min(args.count, len(dataset))):
        image, label = dataset[idx]
        images.append(image)
        labels.append(CIFAR10_CLASSES[label])
    grid = tv_utils.make_grid(torch.stack(images), nrow=4, padding=2)
    output_path = Path(args.output)
    output = ensure_dir(output_path.parent) / output_path.name
    plt.figure(figsize=(7, 7))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title("CIFAR-10 Samples: " + ", ".join(labels[:4]) + " ...")
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    print(f"Saved sample grid to {output}")


if __name__ == "__main__":
    main()
