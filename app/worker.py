"""RQ task functions. The model is loaded lazily once per worker process."""
from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision.models.detection import (
    SSDLite320_MobileNet_V3_Large_Weights,
    ssdlite320_mobilenet_v3_large,
)

_model = None
_weights = None
_model_device = torch.device("cpu")


def resolve_device() -> torch.device:
    """Resolve INFERENCE_DEVICE and fail clearly for an unavailable CUDA request."""
    requested = os.environ.get("INFERENCE_DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda", "auto"}:
        raise RuntimeError("INFERENCE_DEVICE must be one of: cpu, cuda, auto.")
    if requested == "cpu":
        return torch.device("cpu")
    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "INFERENCE_DEVICE=cuda was requested, but CUDA is unavailable. "
            "Use the CPU Compose file or configure the NVIDIA Container Toolkit."
        )
    return torch.device("cuda" if cuda_available else "cpu")


def preload_model() -> None:
    """Load weights/model on CPU before RQ forks job processes."""
    global _model, _weights, _model_device
    if _model is None:
        _weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        _model = ssdlite320_mobilenet_v3_large(weights=_weights)
        _model.eval()
        _model_device = torch.device("cpu")


def get_model(device: torch.device):
    """Return the preloaded model, moving it only within the job process if needed."""
    global _model_device
    preload_model()
    if _model_device != device:
        _model.to(device)
        _model_device = device
    return _model, _weights


def detect(image_filename: str) -> dict:
    """Run one uploaded image through the model and return JSON-safe boxes."""
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
    image_path = upload_dir / image_filename
    if not image_path.is_file():
        raise FileNotFoundError("The uploaded image is no longer available.")

    device = resolve_device()
    model, weights = get_model(device)
    threshold = float(os.environ.get("SCORE_THRESHOLD", "0.5"))
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image_width, image_height = image.size
    tensor = weights.transforms()(image).to(device)

    with torch.inference_mode():
        output = model([tensor])[0]

    categories = weights.meta["categories"]
    detections = []
    for box, label, score in zip(output["boxes"], output["labels"], output["scores"]):
        confidence = float(score.item())
        if confidence < threshold:
            continue
        x1, y1, x2, y2 = (round(float(value), 2) for value in box.detach().cpu().tolist())
        detections.append({
            "label": categories[int(label.item())], "score": round(confidence, 3),
            "box": [x1, y1, x2, y2],
        })
    return {"image_width": image_width, "image_height": image_height, "detections": detections}
