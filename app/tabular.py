"""Dataset preview, ANN training, and latest-model prediction helpers."""
from __future__ import annotations

import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
from torch import nn

from app.worker import resolve_device

MAX_PREVIEW_ROWS = 20
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 64


def dataset_dir() -> Path:
    path = Path(os.environ.get("TABULAR_DATASET_DIR", "data/tabular/datasets"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir() -> Path:
    path = Path(os.environ.get("TABULAR_MODEL_DIR", "data/tabular/models"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_path(dataset_id: str) -> Path | None:
    matches = list(dataset_dir().glob(f"{dataset_id}.*"))
    return next((path for path in matches if path.suffix.lower() in {".csv", ".xlsx", ".xls"}), None)


def sheet_names(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        return ["CSV"]
    return list(pd.ExcelFile(path).sheet_names)


def read_frame(path: Path, sheet: str | None = None) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        if sheet not in (None, "CSV"):
            raise ValueError("CSV files have one sheet named CSV.")
        return pd.read_csv(path)
    names = sheet_names(path)
    selected = sheet or names[0]
    if selected not in names:
        raise ValueError("Selected worksheet was not found.")
    return pd.read_excel(path, sheet_name=selected)


def preview_dataset(path: Path, sheet: str | None = None) -> dict:
    names = sheet_names(path)
    selected = sheet or names[0]
    frame = read_frame(path, selected)
    if frame.empty:
        raise ValueError("The selected worksheet has no data rows.")
    if frame.columns.duplicated().any():
        raise ValueError("Column names must be unique.")
    preview = json.loads(frame.head(MAX_PREVIEW_ROWS).to_json(orient="records", date_format="iso"))
    columns = [
        {"name": str(name), "dtype": str(frame[name].dtype), "numeric": bool(pd.api.types.is_numeric_dtype(frame[name]))}
        for name in frame.columns
    ]
    return {
        "sheet_names": names,
        "selected_sheet": selected,
        "rows": int(len(frame)),
        "columns": columns,
        "preview": preview,
    }


class AnnModel(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.layers(values)


def _latest_pointer() -> Path:
    return model_dir() / "latest.json"


def latest_metadata() -> dict | None:
    pointer = _latest_pointer()
    if not pointer.is_file():
        return None
    return json.loads(pointer.read_text())


def train_tabular_model(filename: str, source_filename: str, sheet: str, target: str, task: str) -> dict:
    """RQ task that creates a durable, self-describing latest ANN artifact."""
    path = dataset_dir() / filename
    frame = read_frame(path, sheet)
    if target not in frame.columns:
        raise ValueError("Selected target column was not found.")
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    features = [str(column) for column in frame.columns if str(column) != target]
    if not features:
        raise ValueError("The dataset needs at least one feature column.")
    if len(frame) < 10:
        raise ValueError("The dataset needs at least 10 rows for training and validation.")
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in features):
        raise ValueError("All feature columns must be numeric.")
    if frame[features].isna().any().any() or frame[target].isna().any():
        raise ValueError("Feature and target values cannot be missing.")

    values = torch.tensor(frame[features].to_numpy(dtype="float32"))
    if not torch.isfinite(values).all():
        raise ValueError("Feature values must be finite numbers.")
    classes: list[str] | None = None
    if task == "classification":
        labels = frame[target].astype(str)
        classes = sorted(labels.unique().tolist())
        if len(classes) < 2:
            raise ValueError("Classification requires at least two target classes.")
        mapping = {label: index for index, label in enumerate(classes)}
        targets = torch.tensor([mapping[label] for label in labels], dtype=torch.long)
        output_size = len(classes)
        criterion: nn.Module = nn.CrossEntropyLoss()
    else:
        numeric_target = pd.to_numeric(frame[target], errors="coerce")
        if numeric_target.isna().any():
            raise ValueError("Regression targets must be numeric.")
        targets = torch.tensor(numeric_target.to_numpy(dtype="float32")).unsqueeze(1)
        output_size = 1
        criterion = nn.MSELoss()

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(values), generator=generator)
    validation_count = max(1, round(len(values) * 0.2))
    validation_indices, train_indices = indices[:validation_count], indices[validation_count:]
    if not len(train_indices):
        raise ValueError("The dataset does not leave rows for training.")
    train_values, validation_values = values[train_indices], values[validation_indices]
    train_targets, validation_targets = targets[train_indices], targets[validation_indices]
    mean = train_values.mean(dim=0)
    std = train_values.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_values = (train_values - mean) / std
    validation_values = (validation_values - mean) / std

    device = resolve_device()
    model = AnnModel(len(features), output_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_state, best_loss, stale_epochs = None, float("inf"), 0
    for epoch in range(MAX_EPOCHS):
        model.train()
        order = torch.randperm(len(train_values), generator=generator)
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            prediction = model(train_values[batch].to(device))
            loss = criterion(prediction, train_targets[batch].to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_loss = float(criterion(model(validation_values.to(device)), validation_targets.to(device)).item())
        if validation_loss < best_loss:
            best_loss, stale_epochs = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    with torch.inference_mode():
        outputs = model(validation_values.to(device)).cpu()
    metrics: dict[str, float] = {"validation_loss": round(best_loss, 6)}
    if task == "classification":
        metrics["validation_accuracy"] = round(float((outputs.argmax(1) == validation_targets).float().mean().item()), 4)
    else:
        errors = outputs.squeeze(1) - validation_targets.squeeze(1)
        metrics["validation_mae"] = round(float(errors.abs().mean().item()), 6)
        metrics["validation_rmse"] = round(float(errors.square().mean().sqrt().item()), 6)

    run_id = str(uuid4())
    run_dir = model_dir() / "runs" / run_id
    run_dir.mkdir(parents=True)
    torch.save(best_state, run_dir / "model.pt")
    source_copy = run_dir / f"source{path.suffix.lower()}"
    shutil.copy2(path, source_copy)
    metadata = {
        "run_id": run_id, "source_filename": source_filename, "source_copy": str(source_copy), "sheet": sheet,
        "target": target, "task": task, "feature_columns": features, "classes": classes,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()}, "architecture": [len(features), 128, 64, output_size],
        "training": {"optimizer": "Adam", "learning_rate": 0.001, "max_epochs": MAX_EPOCHS, "epochs_completed": epoch + 1, "validation_split": 0.2, "batch_size": BATCH_SIZE, "early_stopping_patience": PATIENCE},
        "metrics": metrics, "trained_at": datetime.now(timezone.utc).isoformat(), "model_path": str(run_dir / "model.pt"),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    temporary_pointer = model_dir() / "latest.tmp"
    temporary_pointer.write_text(json.dumps(metadata))
    os.replace(temporary_pointer, _latest_pointer())
    return metadata


def predict_latest(features: dict[str, object]) -> dict:
    metadata = latest_metadata()
    if metadata is None:
        raise FileNotFoundError("No trained ANN model is available.")
    expected = metadata["feature_columns"]
    if set(features) != set(expected):
        missing = sorted(set(expected) - set(features))
        extra = sorted(set(features) - set(expected))
        raise ValueError(f"Features must match the training schema. Missing: {missing}; extra: {extra}.")
    try:
        values = [float(features[column]) for column in expected]
    except (TypeError, ValueError) as error:
        raise ValueError("All prediction feature values must be numeric.") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Prediction feature values must be finite.")
    model = AnnModel(len(expected), metadata["architecture"][-1])
    model.load_state_dict(torch.load(metadata["model_path"], map_location="cpu", weights_only=True))
    model.eval()
    mean, std = torch.tensor(metadata["normalization"]["mean"]), torch.tensor(metadata["normalization"]["std"])
    with torch.inference_mode():
        output = model((torch.tensor(values).unsqueeze(0) - mean) / std).squeeze(0)
    if metadata["task"] == "regression":
        return {"target": metadata["target"], "task": "regression", "prediction": round(float(output.item()), 6)}
    probabilities = output.softmax(0).tolist()
    classes = metadata["classes"]
    return {"target": metadata["target"], "task": "classification", "prediction": classes[int(output.argmax().item())], "probabilities": {label: round(float(score), 6) for label, score in zip(classes, probabilities)}}
