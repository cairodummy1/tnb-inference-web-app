from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


def test_frontend_stops_for_terminal_states_and_has_timeout():
    source = Path("app/static/app.js").read_text()
    assert '"failed", "stopped", "canceled"' in source
    assert "POLL_TIMEOUT_MS" in source


def test_tnb_branding_and_public_assets_are_available():
    client = TestClient(app)
    viewer = client.get("/")
    inference = client.get("/inference")

    assert app.title == "TNB Inference"
    assert viewer.status_code == 200
    assert inference.status_code == 200
    assert "Image Viewer" in viewer.text
    assert "AI Inference" in inference.text
    assert "VisionLab" not in viewer.text
    assert "COCO Object Detector" not in inference.text
    for asset in ("logo.png", "icon.png", "favicon.ico"):
        response = client.get(f"/public/{asset}")
        assert response.status_code == 200


def test_frontend_includes_classification_and_ann_workflows():
    source = Path("app/static/inference.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert "ResNet-50" in source
    assert "ImageNet-1K" in source
    assert "Train from CSV or Excel" in source
    assert "/classification/jobs" in script
    assert "/tabular/training-jobs" in script
    assert "/tabular/predictions" in script


def test_frontend_includes_voc_modal_with_optional_bbox_overlay():
    source = Path("app/static/index.html").read_text()
    script = Path("app/static/viewer.js").read_text()

    assert "Image Viewer" in source
    assert "All objects" in source
    assert 'id="viewer-modal"' in source
    assert "Show bounding boxes &amp; labels" in source
    assert 'class="modal-close"' in source
    assert "viewer-canvas" in source
    assert "/viewer/status" in script
    assert "/viewer/images" in script
    assert "viewerBoxesVisible" in script
    assert 'canvas.removeAttribute("hidden")' in script
    assert "hideViewerBoxes" in script
    assert "selectViewerImage(item)" in script
    assert "/viewer/images/${encodeURIComponent(image.image_id)}" in script
    assert "viewerModal.showModal()" in script
    assert "viewer-download-progress" in source
    assert "Downloading ${download.artifact}" in script
    assert "viewer-start-import" in source
    assert "/viewer/import-jobs" in script
    assert "coco" not in source.lower()
    assert "coco" not in script.lower()
    assert "z-index: 1" in Path("app/static/style.css").read_text()
    assert "canvas[hidden]" in Path("app/static/style.css").read_text()


def test_frontend_uses_accessible_workflow_tabs():
    source = Path("app/static/inference.html").read_text()
    script = Path("app/static/app.js").read_text()

    assert 'role="tablist"' in source
    assert 'role="tabpanel"' in source
    assert "activateTab" in script
    assert "ArrowRight" in script
    assert 'data-tab="coco"' not in source
    assert source.count('role="tab"') == 3
    assert "AI INFERENCE" in source
    assert "coco" not in source.lower()


def test_pages_use_separate_clients_and_active_navigation():
    viewer = Path("app/static/index.html").read_text()
    inference = Path("app/static/inference.html").read_text()

    assert "/static/viewer.js" in viewer
    assert "/static/app.js" not in viewer
    assert "/static/app.js" in inference
    assert "/static/viewer.js" not in inference
    assert 'href="/" aria-current="page"' in viewer
    assert 'href="/inference" aria-current="page"' in inference
