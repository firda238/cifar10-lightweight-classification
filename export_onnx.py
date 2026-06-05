from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models import build_model
from utils import ensure_dir, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export a PyTorch checkpoint to ONNX.")
    parser.add_argument("--model", default="lightweight", choices=["baseline", "lightweight"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/lightweight_cnn.onnx")
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")
    model = build_model(args.model).to(device)
    load_checkpoint(args.checkpoint, model, device)
    model.eval()
    dummy = torch.randn(1, 3, 32, 32, device=device)
    output = Path(args.output)
    output_path = ensure_dir(output.parent) / output.name
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset,
        dynamo=False,
    )
    print(f"Exported ONNX model to {output_path}")


if __name__ == "__main__":
    main()
