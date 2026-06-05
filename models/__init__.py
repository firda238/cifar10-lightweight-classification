from .cnn_baseline import BaselineCNN
from .lightweight_cnn import LightweightCNN


def build_model(name: str, num_classes: int = 10):
    name = name.lower()
    if name in {"baseline", "cnn", "cnn_baseline"}:
        return BaselineCNN(num_classes=num_classes)
    if name in {"lightweight", "lightweight_cnn", "mobile"}:
        return LightweightCNN(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {name}")
