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


def test_frontend_includes_coco_modal_with_optional_bbox_overlay():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert "Image Viewer" in source
    assert "All objects" in source
    assert 'id="coco-modal"' in source
    assert "Show bounding boxes &amp; labels" in source
    assert "coco-canvas" in source
    assert "/coco/status" in script
    assert "/coco/images" in script
    assert "cocoBoxesVisible" in script
    assert "selectCocoImage(item)" in script
    assert "/coco/images/${encodeURIComponent(image.image_key)}" in script
    assert "cocoModal.showModal()" in script
    assert "coco-download-progress" in source
    assert "Downloading ${download.artifact}" in script
    assert "coco-start-import" in source
    assert "/coco/import-jobs" in script


def test_frontend_uses_accessible_workflow_tabs():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert 'role="tablist"' in source
    assert 'role="tabpanel"' in source
    assert "activateTab" in script
    assert "ArrowRight" in script
    assert 'data-tab="coco"' not in source
    assert source.count('role="tab"') == 3
    assert "COCO Dataset" in source
    assert "AI INFERENCE" in source
