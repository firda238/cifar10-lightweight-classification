from __future__ import annotations

import argparse

import torch
from torch import nn

from data_utils import CIFAR10_CLASSES, build_loaders
from models import build_model
from train import choose_device
from utils import benchmark_inference, count_parameters, evaluate, load_checkpoint, plot_confusion, write_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model.")
    parser.add_argument("--model", default="lightweight", choices=["baseline", "lightweight"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--limit-test", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = choose_device(args.device)
    _, test_loader = build_loaders(
        data_dir=args.data_dir,
        augment="none",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
        limit_test=args.limit_test,
    )
    model = build_model(args.model).to(device)
    metadata = load_checkpoint(args.checkpoint, model, device)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    bench = benchmark_inference(model, device)
    summary = {
        "model": args.model,
        "checkpoint": args.checkpoint,
        "checkpoint_metadata": metadata,
        "accuracy": metrics["accuracy"],
        "loss": metrics["loss"],
        "params": count_parameters(model),
        **bench,
    }
    plot_confusion(
        metrics["targets"],
        metrics["preds"],
        CIFAR10_CLASSES,
        f"{args.results_dir}/confusion_matrix_{args.model}.png",
    )
    write_metrics(f"{args.results_dir}/test_{args.model}_metrics.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
