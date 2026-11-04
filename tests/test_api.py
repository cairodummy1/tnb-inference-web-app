import io

from fastapi.testclient import TestClient
from PIL import Image
from redis.exceptions import ConnectionError as RedisConnectionError
from rq.exceptions import NoSuchJobError

from app import main


def image_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_upload_requires_available_worker(monkeypatch):
    monkeypatch.setattr(main, "has_inference_worker", lambda _redis: False)
    response = TestClient(main.app).post("/jobs", files={"image": ("sample.png", image_bytes(), "image/png")})
    assert response.status_code == 503
    assert "No inference worker" in response.json()["detail"]


def test_upload_queues_a_valid_image(monkeypatch, tmp_path):
    queued = {}

    def enqueue(_function, filename, **kwargs):
        queued.update(filename=filename, **kwargs)

    monkeypatch.setattr(main, "upload_dir", tmp_path)
    monkeypatch.setattr(main, "has_inference_worker", lambda _redis: True)
    monkeypatch.setattr(main.queue, "enqueue", enqueue)
    response = TestClient(main.app).post("/jobs", files={"image": ("sample.png", image_bytes(), "image/png")})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert queued["job_id"] == body["job_id"]
    assert (tmp_path / queued["filename"]).is_file()


def test_missing_job_is_404_and_redis_error_is_503(monkeypatch):
    monkeypatch.setattr(main.Job, "fetch", lambda *_args, **_kwargs: (_ for _ in ()).throw(NoSuchJobError("missing")))
    assert TestClient(main.app).get("/jobs/missing").status_code == 404

    monkeypatch.setattr(main.Job, "fetch", lambda *_args, **_kwargs: (_ for _ in ()).throw(RedisConnectionError("down")))
    assert TestClient(main.app).get("/jobs/any").status_code == 503
