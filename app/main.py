from __future__ import annotations

import io
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from redis.exceptions import RedisError
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from app.queueing import has_inference_worker, redis_connection
from app.worker import detect

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
upload_dir.mkdir(parents=True, exist_ok=True)
redis = redis_connection()
queue = Queue("inference", connection=redis, default_timeout=300)

app = FastAPI(title="Simple Object Detection")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> dict:
    try:
        redis.ping()
        worker_ready = has_inference_worker(redis)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    if not worker_ready:
        raise HTTPException(status_code=503, detail="No inference worker is available")
    return {"status": "ok", "worker": "ready"}


@app.post("/jobs", status_code=202)
async def create_job(image: UploadFile = File(...)) -> dict:
    try:
        if not has_inference_worker(redis):
            raise HTTPException(status_code=503, detail="No inference worker is available. Try again shortly.")
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    content = await image.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Please choose an image.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Images must be 10 MB or smaller.")

    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(content)) as candidate:
            candidate.verify()
        with Image.open(io.BytesIO(content)) as candidate:
            image_format = candidate.format
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Upload a valid JPEG, PNG, or WebP image.") from error

    extension = ALLOWED_FORMATS.get(image_format or "")
    if not extension:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")
    job_id = str(uuid4())
    filename = f"{job_id}.{extension}"
    (upload_dir / filename).write_bytes(content)
    try:
        queue.enqueue(detect, filename, job_id=job_id, result_ttl=3600, failure_ttl=86400)
    except RedisError as error:
        (upload_dir / filename).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not queue the inference job.") from error
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        job = Job.fetch(job_id, connection=redis)
        status = job.get_status(refresh=True)
    except NoSuchJobError as error:
        raise HTTPException(status_code=404, detail="Job not found or it has expired.") from error
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    # RQ 1.x returns a string; newer versions can return a JobStatus enum.
    status_value = getattr(status, "value", status)
    response = {"job_id": job.id, "status": status_value}
    if status_value == "finished":
        response["result"] = job.result
    elif status_value == "failed":
        response["error"] = "Inference failed. Check the worker logs for details."
    return response
