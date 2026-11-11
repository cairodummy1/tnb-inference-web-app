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
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import nms

_model = None
_weights = None
_model_device = torch.device("cpu")
_classifier = None
_classifier_weights = None
_classifier_device = torch.device("cpu")


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
    if cuda_available:
        major, minor = torch.cuda.get_device_capability()
        required_arch = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and required_arch not in supported_arches and (
            f"compute_{major}{minor}" not in supported_arches
        ):
            raise RuntimeError(
                f"This PyTorch CUDA build does not support {required_arch}. "
                "Use a PyTorch build compiled for this GPU; Jetson Orin users can use "
                "compose.jetson.yaml."
            )
    return torch.device("cuda" if cuda_available else "cpu")


def validate_cuda_runtime(device: torch.device) -> None:
    """Verify CUDA tensor execution and torchvision's CUDA NMS before serving jobs."""
    if device.type != "cuda":
        return
    try:
        boxes = torch.tensor([[0, 0, 2, 2], [0, 0, 1, 1]], device=device, dtype=torch.float32)
        scores = torch.tensor([0.9, 0.8], device=device)
        nms(boxes, scores, 0.5)
        torch.cuda.synchronize(device)
    except (RuntimeError, NotImplementedError) as error:
        raise RuntimeError(
            "CUDA is visible, but PyTorch/torchvision could not execute the CUDA NMS operation. "
            "Use a Jetson-compatible PyTorch and torchvision image."
        ) from error


def preload_model() -> None:
    """Load public image models on CPU before RQ starts processing jobs."""
    preload_detector()
    preload_classifier()


def preload_detector() -> None:
    """Load the detector without forcing classification weights to download."""
    global _model, _weights, _model_device
    if _model is None:
        _weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        _model = ssdlite320_mobilenet_v3_large(weights=_weights)
        _model.eval()
        _model_device = torch.device("cpu")

def preload_classifier() -> None:
    """Load the classifier without forcing detection weights to download."""
    global _classifier, _classifier_weights, _classifier_device
    if _classifier is None:
        _classifier_weights = ResNet50_Weights.DEFAULT
        _classifier = resnet50(weights=_classifier_weights)
        _classifier.eval()
        _classifier_device = torch.device("cpu")


def get_model(device: torch.device):
    """Return the preloaded model, moving it only within the job process if needed."""
    global _model_device
    preload_detector()
    if _model_device != device:
        _model.to(device)
        _model_device = device
    return _model, _weights


def get_classifier(device: torch.device):
    """Return ResNet-50 and move it to the selected device when required."""
    global _classifier_device
    preload_classifier()
    if _classifier_device != device:
        _classifier.to(device)
        _classifier_device = device
    return _classifier, _classifier_weights


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


def classify(image_filename: str) -> dict:
    """Classify an uploaded image with ImageNet-pretrained ResNet-50."""
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
    image_path = upload_dir / image_filename
    if not image_path.is_file():
        raise FileNotFoundError("The uploaded image is no longer available.")

    device = resolve_device()
    model, weights = get_classifier(device)
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).squeeze(0).softmax(0)
    scores, indices = probabilities.topk(5)
    categories = weights.meta["categories"]
    predictions = [
        {"label": categories[int(index.item())], "score": round(float(score.item()), 4)}
        for score, index in zip(scores.detach().cpu(), indices.detach().cpu())
    ]
    return {"model": "ResNet-50 ImageNet-1K", "predictions": predictions}
