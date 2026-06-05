from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt

from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Render model_compare.csv as a PNG table.")
    parser.add_argument("--csv", default="results/model_compare.csv")
    parser.add_argument("--output", default="results/model_compare_table.png")
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing comparison CSV: {csv_path}")
    with csv_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Comparison CSV has no rows")

    columns = ["model", "augment", "accuracy", "params", "model_size_mb", "inference_ms", "fps"]
    table_data = []
    for row in rows:
        table_data.append(
            [
                row.get("model", ""),
                row.get("augment", ""),
                f"{float(row.get('accuracy') or 0):.4f}",
                row.get("params", ""),
                f"{float(row.get('model_size_mb') or 0):.2f}",
                f"{float(row.get('inference_ms') or 0):.3f}",
                f"{float(row.get('fps') or 0):.1f}",
            ]
        )

    output = Path(args.output)
    ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=(11, max(2.5, len(table_data) * 0.55 + 1.2)))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Saved comparison table to {output}")


if __name__ == "__main__":
    main()
