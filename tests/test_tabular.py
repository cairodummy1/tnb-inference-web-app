import json

from app import tabular


def test_preview_and_train_regression_model(monkeypatch, tmp_path):
    monkeypatch.setenv("TABULAR_DATASET_DIR", str(tmp_path / "datasets"))
    monkeypatch.setenv("TABULAR_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    dataset_dir = tabular.dataset_dir()
    path = dataset_dir / "sample.csv"
    path.write_text("feature_a,feature_b,target\n" + "\n".join(f"{index},{index * 2},{index * 3}" for index in range(12)))

    preview = tabular.preview_dataset(path)
    assert preview["rows"] == 12
    assert [column["name"] for column in preview["columns"]] == ["feature_a", "feature_b", "target"]

    metadata = tabular.train_tabular_model(path.name, "training.csv", "CSV", "target", "regression")
    assert metadata["target"] == "target"
    assert metadata["task"] == "regression"
    assert json.loads((tabular.model_dir() / "latest.json").read_text())["run_id"] == metadata["run_id"]
    result = tabular.predict_latest({"feature_a": 2, "feature_b": 4})
    assert result["target"] == "target"
    assert result["task"] == "regression"


def test_tabular_prediction_requires_the_training_schema(monkeypatch, tmp_path):
    monkeypatch.setenv("TABULAR_MODEL_DIR", str(tmp_path / "models"))
    tabular.model_dir().mkdir(parents=True, exist_ok=True)
    try:
        tabular.predict_latest({"unknown": 1})
    except FileNotFoundError as error:
        assert "No trained ANN" in str(error)
    else:
        raise AssertionError("Expected a missing-model error")
