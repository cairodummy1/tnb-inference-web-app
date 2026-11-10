from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from redis.exceptions import RedisError
from rq import Queue
from rq.job import Job
from rq.exceptions import NoSuchJobError

from app.queueing import has_coco_worker, has_inference_worker, has_training_worker, has_viewer_worker, redis_connection
from app.coco import database as coco_database, image_details as coco_image_details, image_root as coco_image_root, import_coco2017, list_images as coco_list_images, status as coco_status
from app import viewer
from app.tabular import dataset_path, preview_dataset, predict_latest, train_tabular_model
from app.worker import classify, detect

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
upload_dir.mkdir(parents=True, exist_ok=True)
redis = redis_connection()
queue = Queue("inference", connection=redis, default_timeout=300)
training_queue = Queue("training", connection=redis, default_timeout=3600)
coco_queue = Queue("coco-import", connection=redis, default_timeout=86400)
viewer_queue = Queue("viewer-import", connection=redis, default_timeout=86400)
MAX_VIEWER_ARCHIVE_BYTES = int(os.environ.get("VIEWER_MAX_ARCHIVE_BYTES", str(2 * 1024 * 1024 * 1024)))

app = FastAPI(title="TNB Inference")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.mount("/public", StaticFiles(directory="app/public"), name="public")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/inference", include_in_schema=False)
def inference_page() -> FileResponse:
    return FileResponse("app/static/inference.html")


@app.get("/health")
def health() -> dict:
    try:
        redis.ping()
        worker_ready = has_inference_worker(redis)
        training_ready = has_training_worker(redis)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    if not worker_ready:
        raise HTTPException(status_code=503, detail="No inference worker is available")
    return {"status": "ok", "worker": "ready", "training_worker": "ready" if training_ready else "unavailable"}


async def save_image_upload(image: UploadFile) -> str:
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
    filename = f"{uuid4()}.{extension}"
    (upload_dir / filename).write_bytes(content)
    return filename


@app.post("/jobs", status_code=202)
async def create_job(image: UploadFile = File(...)) -> dict:
    try:
        if not has_inference_worker(redis):
            raise HTTPException(status_code=503, detail="No inference worker is available. Try again shortly.")
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    filename = await save_image_upload(image)
    job_id = Path(filename).stem
    try:
        queue.enqueue(detect, filename, job_id=job_id, result_ttl=3600, failure_ttl=86400)
    except RedisError as error:
        (upload_dir / filename).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not queue the inference job.") from error
    return {"job_id": job_id, "status": "queued"}


@app.post("/classification/jobs", status_code=202)
async def create_classification_job(image: UploadFile = File(...)) -> dict:
    try:
        if not has_inference_worker(redis):
            raise HTTPException(status_code=503, detail="No inference worker is available. Try again shortly.")
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    filename = await save_image_upload(image)
    job_id = Path(filename).stem
    try:
        queue.enqueue(classify, filename, job_id=job_id, result_ttl=3600, failure_ttl=86400)
    except RedisError as error:
        (upload_dir / filename).unlink(missing_ok=True)
        raise HTTPException(status_code=503, detail="Could not queue the classification job.") from error
    return {"job_id": job_id, "status": "queued"}


class TrainingRequest(BaseModel):
    dataset_id: str
    sheet: str
    target: str
    task: str


class PredictionRequest(BaseModel):
    features: dict[str, object]


@app.post("/tabular/datasets", status_code=201)
async def upload_tabular_dataset(dataset: UploadFile = File(...)) -> dict:
    suffix = Path(dataset.filename or "").suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel (.xlsx, .xls) file.")
    content = await dataset.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="Please choose a dataset file.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Dataset files must be 10 MB or smaller.")
    from app.tabular import dataset_dir
    dataset_id = str(uuid4())
    saved = dataset_dir() / f"{dataset_id}{suffix}"
    saved.write_bytes(content)
    try:
        preview = preview_dataset(saved)
    except (ValueError, OSError, UnicodeDecodeError) as error:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read the dataset: {error}") from error
    (dataset_dir() / f"{dataset_id}.json").write_text(json.dumps({"filename": dataset.filename or saved.name}))
    return {"dataset_id": dataset_id, "filename": dataset.filename, **preview}


@app.get("/tabular/datasets/{dataset_id}/preview")
def get_dataset_preview(dataset_id: str, sheet: str | None = None) -> dict:
    path = dataset_path(dataset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or it has expired.")
    try:
        return preview_dataset(path, sheet)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/tabular/training-jobs", status_code=202)
def create_training_job(request: TrainingRequest) -> dict:
    path = dataset_path(request.dataset_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Dataset not found or it has expired.")
    if request.task not in {"classification", "regression"}:
        raise HTTPException(status_code=400, detail="Task must be classification or regression.")
    try:
        preview = preview_dataset(path, request.sheet)
        if request.target not in {column["name"] for column in preview["columns"]}:
            raise ValueError("Select a target column from the dataset preview.")
        if not has_training_worker(redis):
            raise HTTPException(status_code=503, detail="No training worker is available. Try again shortly.")
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Redis is unavailable") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    original_name = json.loads((path.parent / f"{request.dataset_id}.json").read_text())["filename"]
    job_id = str(uuid4())
    try:
        training_queue.enqueue(train_tabular_model, path.name, original_name, request.sheet, request.target, request.task, job_id=job_id, result_ttl=3600, failure_ttl=86400)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Could not queue the training job.") from error
    return {"job_id": job_id, "status": "queued"}


@app.get("/tabular/model")
def get_tabular_model() -> dict:
    from app.tabular import latest_metadata
    metadata = latest_metadata()
    if metadata is None:
        raise HTTPException(status_code=404, detail="No ANN model has been trained yet.")
    return metadata


@app.post("/tabular/predictions")
def create_tabular_prediction(request: PredictionRequest) -> dict:
    try:
        return predict_latest(request.features)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def get_coco_database():
    try:
        db = coco_database()
        db.command("ping")
        return db
    except Exception as error:
        raise HTTPException(status_code=503, detail="COCO catalog database is unavailable.") from error


def get_viewer_database():
    try:
        db = viewer.database()
        db.command("ping")
        return db
    except Exception as error:
        raise HTTPException(status_code=503, detail="Image viewer catalog database is unavailable.") from error


def enqueue_viewer_import(function, *args) -> dict:
    db = get_viewer_database()
    current = viewer.status(db)
    if current["status"] in {"queued", "running"}:
        raise HTTPException(status_code=409, detail="An image viewer import is already queued or running.")
    try:
        if not has_viewer_worker(redis):
            raise HTTPException(status_code=503, detail="No image viewer import worker is available. Try again shortly.")
        job = viewer_queue.enqueue(function, *args, result_ttl=604800, failure_ttl=604800)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Could not queue the image viewer import.") from error
    db.viewer_imports.update_one(
        {"_id": "active"},
        {"$set": {"status": "queued", "job_id": job.id, "error": None, "download": None, "updated_at": datetime.now(timezone.utc).isoformat()}},
        upsert=True,
    )
    return {"job_id": job.id, "status": "queued"}


@app.get("/viewer/status")
def get_viewer_status() -> dict:
    return viewer.status(get_viewer_database())


@app.post("/viewer/import-jobs", status_code=202)
def create_default_viewer_import_job() -> dict:
    return enqueue_viewer_import(viewer.import_default_dataset)


@app.post("/viewer/datasets", status_code=202)
async def upload_viewer_dataset(dataset: UploadFile = File(...)) -> dict:
    if Path(dataset.filename or "").suffix.lower() != ".zip":
        raise HTTPException(status_code=400, detail="Upload a Pascal VOC dataset ZIP file.")
    filename = f"{uuid4().hex}.zip"
    saved = viewer.upload_dir() / filename
    size = 0
    try:
        with saved.open("wb") as output:
            while chunk := await dataset.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_VIEWER_ARCHIVE_BYTES:
                    raise HTTPException(status_code=413, detail="Dataset archive is too large.")
                output.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail="Please choose a dataset archive.")
        with zipfile.ZipFile(saved) as archive:
            viewer._safe_members(archive)
    except (zipfile.BadZipFile, ValueError) as error:
        saved.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid dataset archive: {error}") from error
    except HTTPException:
        saved.unlink(missing_ok=True)
        raise
    return enqueue_viewer_import(viewer.import_uploaded_dataset, filename, dataset.filename or "Custom VOC dataset")


@app.get("/viewer/images")
def get_viewer_images(
    category_id: int | None = Query(default=None, ge=1),
    cursor: str | None = None,
    limit: int = Query(default=24, ge=1, le=100),
) -> dict:
    return viewer.list_images(get_viewer_database(), category_id, cursor, limit)


@app.get("/viewer/images/{image_id}")
def get_viewer_image(image_id: str) -> dict:
    result = viewer.image_details(get_viewer_database(), image_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Image was not found.")
    return result


@app.get("/viewer/images/{image_id}/file")
def get_viewer_image_file(image_id: str) -> FileResponse:
    path = viewer.image_path(get_viewer_database(), image_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Image file is unavailable on this server.")
    return FileResponse(path)


@app.get("/coco/status")
def get_coco_status() -> dict:
    return coco_status(get_coco_database())


@app.post("/coco/import-jobs", status_code=202)
def create_coco_import_job() -> dict:
    db = get_coco_database()
    current = coco_status(db)
    if current["status"] in {"queued", "running"} and current.get("job_id"):
        try:
            job = Job.fetch(current["job_id"], connection=redis)
            current_status = job.get_status(refresh=True)
            job_status = getattr(current_status, "value", current_status)
        except (NoSuchJobError, RedisError):
            job_status = ""
        if job_status in {"queued", "started", "deferred", "scheduled"}:
            raise HTTPException(status_code=409, detail="A COCO import job is already queued or running.")
    if current["status"] == "ready":
        raise HTTPException(status_code=409, detail="COCO is already imported.")
    try:
        if not has_coco_worker(redis):
            raise HTTPException(status_code=503, detail="No COCO import worker is available. Try again shortly.")
        job = coco_queue.enqueue(import_coco2017, result_ttl=604800, failure_ttl=604800)
    except RedisError as error:
        raise HTTPException(status_code=503, detail="Could not queue the COCO import job.") from error
    db.coco_imports.update_one({"_id": "coco2017"}, {"$set": {"status": "queued", "job_id": job.id, "queued_at": datetime.now(timezone.utc).isoformat(), "error": None}, "$unset": {"download": ""}}, upsert=True)
    return {"job_id": job.id, "status": "queued"}


@app.get("/coco/images")
def get_coco_images(
    split: str | None = Query(default=None, pattern="^(train2017|val2017)$"),
    category_id: int | None = Query(default=None, ge=1),
    cursor: str | None = None,
    limit: int = Query(default=24, ge=1, le=100),
) -> dict:
    db = get_coco_database()
    current = coco_status(db)
    if current["status"] != "ready":
        raise HTTPException(status_code=409, detail="COCO data has not finished importing.")
    return coco_list_images(db, split, category_id, cursor, limit)


@app.get("/coco/images/{image_key}")
def get_coco_image(image_key: str) -> dict:
    result = coco_image_details(get_coco_database(), image_key)
    if result is None:
        raise HTTPException(status_code=404, detail="COCO image was not found.")
    return result


@app.get("/coco/images/{image_key}/file")
def get_coco_image_file(image_key: str) -> FileResponse:
    try:
        split, raw_id = image_key.split(":", 1)
        if split not in {"train2017", "val2017"}:
            raise ValueError
        coco_id = int(raw_id)
    except ValueError as error:
        raise HTTPException(status_code=404, detail="COCO image was not found.") from error
    image = get_coco_database().coco_images.find_one({"split": split, "coco_id": coco_id}, {"filename": 1})
    if image is None:
        raise HTTPException(status_code=404, detail="COCO image was not found.")
    path = coco_image_root() / split / image["filename"]
    if not path.is_file():
        raise HTTPException(status_code=404, detail="COCO image file is unavailable on this server.")
    return FileResponse(path, media_type="image/jpeg")


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
        response["error"] = "Job failed. Check the worker logs for details."
    return response
