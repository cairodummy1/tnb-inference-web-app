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
