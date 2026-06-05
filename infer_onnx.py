from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run ONNX Runtime inference.")
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", help="Optional image path. Uses random input when omitted.")
    parser.add_argument("--repeat", type=int, default=200)
    return parser.parse_args()


def load_input(image_path: str | None):
    if not image_path:
        return np.random.randn(1, 3, 32, 32).astype(np.float32)
    from PIL import Image
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    image = Image.open(Path(image_path)).convert("RGB")
    return transform(image).unsqueeze(0).numpy().astype(np.float32)


def main():
    args = parse_args()
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        args.onnx,
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    sample = load_input(args.image)
    logits = session.run(None, {input_name: sample})[0]
    pred = int(np.argmax(logits, axis=1)[0])

    for _ in range(20):
        session.run(None, {input_name: sample})
    start = time.perf_counter()
    for _ in range(args.repeat):
        session.run(None, {input_name: sample})
    elapsed = time.perf_counter() - start
    ms = elapsed / args.repeat * 1000
    print(
        {
            "prediction": CIFAR10_CLASSES[pred],
            "class_id": pred,
            "inference_ms": ms,
            "fps": 1000.0 / ms if ms else 0.0,
        }
    )


if __name__ == "__main__":
    main()
