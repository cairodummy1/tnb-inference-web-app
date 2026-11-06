from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_frontend_stops_for_terminal_states_and_has_timeout():
    source = Path("app/static/app.js").read_text()
    assert '"failed", "stopped", "canceled"' in source
    assert "POLL_TIMEOUT_MS" in source


def test_tnb_branding_and_public_assets_are_available():
    client = TestClient(app)
    page = client.get("/")

    assert app.title == "TNB Inference"
    assert page.status_code == 200
    assert "<title>TNB Inference</title>" in page.text
    assert "TNB Inference" in page.text
    assert "VisionLab" not in page.text
    assert "COCO Object Detector" not in page.text
    for asset in ("logo.png", "icon.png", "favicon.ico"):
        response = client.get(f"/public/{asset}")
        assert response.status_code == 200


def test_frontend_includes_classification_and_ann_workflows():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert "ResNet-50" in source
    assert "ImageNet-1K" in source
    assert "Train from CSV or Excel" in source
    assert "/classification/jobs" in script
    assert "/tabular/training-jobs" in script
    assert "/tabular/predictions" in script


def test_frontend_includes_coco_browser_and_bbox_toggle():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert "COCO 2017 image browser" in source
    assert "Show bounding boxes" in source
    assert "/coco/status" in script
    assert "/coco/images" in script
    assert "cocoBoxesVisible" in script
    assert "coco-download-progress" in source
    assert "Downloading ${download.artifact}" in script


def test_frontend_uses_accessible_workflow_tabs():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert 'role="tablist"' in source
    assert 'role="tabpanel"' in source
    assert "activateTab" in script
    assert "ArrowRight" in script
