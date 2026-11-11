import os

import pytest
import torch
from PIL import Image

from app import worker


class FakeWeights:
    meta = {"categories": ["__background__", "person"]}

    @staticmethod
    def transforms():
        return lambda _image: torch.zeros((3, 12, 16))


class FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def __call__(self, _batch):
        return [{
            "boxes": torch.tensor([[1.25, 2.5, 10.0, 11.5]], device=self.device),
            "labels": torch.tensor([1], device=self.device),
            "scores": torch.tensor([0.9], device=self.device),
        }]


class FakeClassifierWeights:
    meta = {"categories": ["zero", "one", "two", "three", "four", "five"]}

    @staticmethod
    def transforms():
        return lambda _image: torch.zeros((3, 12, 16))


class FakeClassifier:
    def __init__(self):
        self.device = torch.device("cpu")

    def to(self, device):
        self.device = device
        return self

    def __call__(self, _batch):
        return torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], device=self.device)


def test_resolve_cpu_and_rejects_unknown_device(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    assert worker.resolve_device().type == "cpu"
    monkeypatch.setenv("INFERENCE_DEVICE", "not-a-device")
    with pytest.raises(RuntimeError, match="cpu, cuda, auto"):
        worker.resolve_device()


def test_explicit_cuda_fails_when_unavailable(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        worker.resolve_device()


def test_explicit_cuda_rejects_an_unsupported_architecture(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: ["sm_90"])
    with pytest.raises(RuntimeError, match="sm_120"):
        worker.resolve_device()


def test_cuda_accepts_an_empty_architecture_list(monkeypatch):
    monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (8, 7))
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: [])
    assert worker.resolve_device().type == "cuda"


def test_validate_cuda_runtime_executes_torchvision_nms(monkeypatch):
    calls = {}
    monkeypatch.setattr(worker.torch, "tensor", lambda values, **kwargs: (values, kwargs))
    monkeypatch.setattr(
        worker,
        "nms",
        lambda boxes, scores, threshold: calls.update(
            boxes=boxes, scores=scores, threshold=threshold
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.update(device=device))

    worker.validate_cuda_runtime(torch.device("cuda"))

    assert calls["boxes"][1]["device"].type == "cuda"
    assert calls["scores"][1]["device"].type == "cuda"
    assert calls["threshold"] == 0.5
    assert calls["device"].type == "cuda"


def test_detect_returns_json_safe_detection(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 12), "white").save(image_path)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setenv("SCORE_THRESHOLD", "0.5")
    monkeypatch.setattr(worker, "_model", FakeModel())
    monkeypatch.setattr(worker, "_weights", FakeWeights())
    monkeypatch.setattr(worker, "_model_device", torch.device("cpu"))

    result = worker.detect(image_path.name)

    assert result == {
        "image_width": 16,
        "image_height": 12,
        "detections": [{"label": "person", "score": 0.9, "box": [1.25, 2.5, 10.0, 11.5]}],
    }


def test_classify_returns_top_five_json_safe_predictions(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (16, 12), "white").save(image_path)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    monkeypatch.setattr(worker, "_classifier", FakeClassifier())
    monkeypatch.setattr(worker, "_classifier_weights", FakeClassifierWeights())
    monkeypatch.setattr(worker, "_classifier_device", torch.device("cpu"))

    result = worker.classify(image_path.name)

    assert result["model"] == "ResNet-50 ImageNet-1K"
    assert [item["label"] for item in result["predictions"]] == ["five", "four", "three", "two", "one"]
