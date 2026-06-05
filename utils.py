from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path: str | Path, model, metadata: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)


def load_checkpoint(path: str | Path, model, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint.get("metadata", {})


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == targets).sum().item()
            total_samples += images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "preds": all_preds,
        "targets": all_targets,
    }


def benchmark_inference(model, device, repeat: int = 200, warmup: int = 30):
    model.eval()
    sample = torch.randn(1, 3, 32, 32, device=device)
    synchronize = getattr(torch.mps, "synchronize", None) if device.type == "mps" else None
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if synchronize:
            synchronize()
        start = time.perf_counter()
        for _ in range(repeat):
            model(sample)
        if synchronize:
            synchronize()
        elapsed = time.perf_counter() - start
    ms = elapsed / repeat * 1000
    return {"inference_ms": ms, "fps": 1000.0 / ms if ms else 0.0}


def write_metrics(path: str | Path, metrics: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def append_compare_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    exists = path.exists()
    fields = [
        "model",
        "augment",
        "accuracy",
        "loss",
        "params",
        "model_size_mb",
        "inference_ms",
        "fps",
    ]
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def plot_curves(history: list[dict], output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_dir = ensure_dir(output_dir)
    epochs = [item["epoch"] for item in history]
    for key, title, filename in [
        ("accuracy", "Accuracy Curve", "accuracy_curve.png"),
        ("loss", "Loss Curve", "loss_curve.png"),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(epochs, [item[f"train_{key}"] for item in history], label=f"train {key}")
        plt.plot(epochs, [item[f"test_{key}"] for item in history], label=f"test {key}")
        plt.xlabel("Epoch")
        plt.ylabel(key.capitalize())
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=160)
        plt.close()


def plot_confusion(targets, preds, labels, path: str | Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    path = Path(path)
    ensure_dir(path.parent)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for target, pred in zip(targets, preds):
        matrix[int(target), int(pred)] += 1

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
                fontsize=7,
            )
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)


def file_size_mb(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)
