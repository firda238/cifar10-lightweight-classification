from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn, optim
from tqdm import tqdm

from data_utils import AUGMENT_CHOICES, CIFAR10_CLASSES, build_loaders, set_seed
from models import build_model
from utils import (
    append_compare_csv,
    benchmark_inference,
    count_parameters,
    ensure_dir,
    evaluate,
    file_size_mb,
    plot_confusion,
    plot_curves,
    save_checkpoint,
    write_metrics,
)


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(name: str, model, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, desc="train", leave=False)
    for images, targets in progress:
        images = images.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        preds = outputs.argmax(dim=1)
        total_loss += loss.item() * images.size(0)
        total_correct += (preds == targets).sum().item()
        total_samples += images.size(0)
        progress.set_postfix(
            loss=f"{total_loss / max(total_samples, 1):.4f}",
            acc=f"{total_correct / max(total_samples, 1):.4f}",
        )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models.")
    parser.add_argument("--model", default="lightweight", choices=["baseline", "lightweight"])
    parser.add_argument(
        "--augment",
        default="rotate",
        choices=AUGMENT_CHOICES,
        help=(
            "Augmentation strategy: none=no augmentation, "
            "rotate=random crop + horizontal flip + random rotation, "
            "noise=random crop + horizontal flip + Gaussian noise, "
            "strong=crop + flip + rotation + color jitter + Gaussian noise."
        ),
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    train_loader, test_loader = build_loaders(
        data_dir=args.data_dir,
        augment=args.augment,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
        limit_train=args.limit_train,
        limit_test=args.limit_test,
    )
    model = build_model(args.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args.optimizer, model, args.lr, args.weight_decay)

    history = []
    best_acc = -1.0
    checkpoint_dir = ensure_dir(args.checkpoint_dir)
    results_dir = ensure_dir(args.results_dir)
    run_dir = ensure_dir(results_dir / f"{args.model}_{args.augment}")
    checkpoint_path = checkpoint_dir / f"{args.model}_{args.augment}.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
        }
        history.append(row)
        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"train_acc={row['train_accuracy']:.4f} "
            f"test_loss={row['test_loss']:.4f} "
            f"test_acc={row['test_accuracy']:.4f}",
            flush=True,
        )
        if test_metrics["accuracy"] > best_acc:
            best_acc = test_metrics["accuracy"]
            save_checkpoint(
                checkpoint_path,
                model,
                {
                    "model": args.model,
                    "augment": args.augment,
                    "accuracy": best_acc,
                    "classes": CIFAR10_CLASSES,
                    "input_shape": [1, 3, 32, 32],
                },
            )

    plot_curves(history, run_dir)
    final_metrics = evaluate(model, test_loader, criterion, device)
    plot_confusion(
        final_metrics["targets"],
        final_metrics["preds"],
        CIFAR10_CLASSES,
        run_dir / "confusion_matrix.png",
    )
    bench = benchmark_inference(model, device)
    summary = {
        "model": args.model,
        "augment": args.augment,
        "epochs": args.epochs,
        "best_accuracy": best_acc,
        "final_accuracy": final_metrics["accuracy"],
        "final_loss": final_metrics["loss"],
        "params": count_parameters(model),
        "checkpoint": str(checkpoint_path),
        "model_size_mb": file_size_mb(checkpoint_path),
        **bench,
    }
    write_metrics(run_dir / "metrics.json", summary)
    append_compare_csv(
        results_dir / "model_compare.csv",
        {
            "model": args.model,
            "augment": args.augment,
            "accuracy": final_metrics["accuracy"],
            "loss": final_metrics["loss"],
            "params": count_parameters(model),
            "model_size_mb": file_size_mb(checkpoint_path),
            **bench,
        },
    )
    print(summary, flush=True)


if __name__ == "__main__":
    main()
