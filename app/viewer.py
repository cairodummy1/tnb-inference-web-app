"""Pascal VOC dataset import and query helpers for the public image viewer."""
from __future__ import annotations

import os
import shutil
import urllib.request
import uuid
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image
from pymongo import ASCENDING, MongoClient

DEFAULT_DATASET_URL = "https://f005.backblazeb2.com/file/jetson-assets/dataset/transmission%20line%20obj%20det.zip"
BATCH_SIZE = 500


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_dir() -> Path:
    path = Path(os.environ.get("VIEWER_DATA_DIR", "data/viewer"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def upload_dir() -> Path:
    path = data_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_root() -> Path:
    path = data_dir() / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database():
    client = MongoClient(os.environ.get("MONGODB_URL", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5_000)
    return client[os.environ.get("VIEWER_DATABASE", "tnb_inference")]


def ensure_indexes(db) -> None:
    db.viewer_images.create_index([("dataset_id", ASCENDING), ("image_id", ASCENDING)], unique=True)
    db.viewer_images.create_index([("dataset_id", ASCENDING), ("category_ids", ASCENDING)])
    db.viewer_datasets.create_index([("dataset_id", ASCENDING)], unique=True)


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Dataset archive contains an unsafe path.")
    return members


def _xml_record(archive: zipfile.ZipFile, member: zipfile.ZipInfo, names: set[str]) -> tuple[str, dict] | None:
    try:
        root = ET.fromstring(archive.read(member))
        filename = (root.findtext("filename") or "").strip()
        width = int(root.findtext("size/width") or "0")
        height = int(root.findtext("size/height") or "0")
    except (ET.ParseError, ValueError) as error:
        raise ValueError(f"Invalid VOC annotation: {member.filename}") from error
    if not filename or width <= 0 or height <= 0:
        raise ValueError(f"VOC annotation is missing filename or image dimensions: {member.filename}")

    annotation_path = PurePosixPath(member.filename)
    image_path = str(annotation_path.parent.parent / "JPEGImages" / filename)
    if image_path not in names:
        return None
    annotations = []
    for object_node in root.findall("object"):
        label = (object_node.findtext("name") or "").strip()
        box = object_node.find("bndbox")
        if not label or box is None:
            continue
        try:
            xmin, ymin = float(box.findtext("xmin") or ""), float(box.findtext("ymin") or "")
            xmax, ymax = float(box.findtext("xmax") or ""), float(box.findtext("ymax") or "")
        except ValueError:
            continue
        xmin, xmax = max(0.0, xmin), min(float(width), xmax)
        ymin, ymax = max(0.0, ymin), min(float(height), ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        annotations.append({"label": label, "bbox": [xmin, ymin, xmax - xmin, ymax - ymin]})
    return image_path, {"filename": filename, "width": width, "height": height, "annotations": annotations}


def _download(url: str, destination: Path, progress) -> None:
    temporary = destination.with_suffix(".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            downloaded += len(chunk)
            progress(downloaded, total, round(downloaded / total * 100, 1) if total else None)
    temporary.replace(destination)


def _set_status(db, **fields) -> None:
    fields["updated_at"] = now()
    db.viewer_imports.update_one({"_id": "active"}, {"$set": fields}, upsert=True)


def _set_failure(db, error: Exception) -> None:
    current = db.viewer_imports.find_one({"_id": "active"}) or {}
    _set_status(
        db,
        status="ready" if current.get("active_dataset") else "failed",
        error=str(error),
        download=None,
    )


def _import_archive(archive_path: Path, source: str) -> None:
    db = database()
    ensure_indexes(db)
    dataset_id = uuid.uuid4().hex
    staging = dataset_root() / f".{dataset_id}.staging"
    final = dataset_root() / dataset_id
    try:
        _set_status(db, status="running", source=source, error=None, download=None)
        staging.mkdir(parents=True, exist_ok=False)
        with zipfile.ZipFile(archive_path) as archive:
            members = _safe_members(archive)
            names = {member.filename for member in members if not member.is_dir()}
            records: list[tuple[str, dict]] = []
            skipped = 0
            for member in members:
                if member.is_dir() or not member.filename.lower().endswith(".xml"):
                    continue
                record = _xml_record(archive, member, names)
                if record is None:
                    skipped += 1
                else:
                    records.append(record)
            if not records:
                raise ValueError("No usable Pascal VOC image/XML pairs were found.")
            labels = sorted({annotation["label"] for _, record in records for annotation in record["annotations"]})
            category_ids = {label: index + 1 for index, label in enumerate(labels)}
            for image_path, _ in records:
                target = staging / image_path
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(image_path) as source_file, target.open("wb") as destination_file:
                    shutil.copyfileobj(source_file, destination_file)

        images = []
        for index, (image_path, record) in enumerate(records, start=1):
            image_id = f"{dataset_id}:{index}"
            relative_path = image_path
            image_file = staging / relative_path
            try:
                with Image.open(image_file) as image:
                    image.verify()
            except (OSError, Image.DecompressionBombError) as error:
                raise ValueError(f"Invalid image referenced by VOC annotation: {record['filename']}") from error
            annotations = [{**annotation, "category_id": category_ids[annotation["label"]]} for annotation in record["annotations"]]
            images.append({
                "image_id": image_id, "dataset_id": dataset_id, "filename": record["filename"],
                "width": record["width"], "height": record["height"], "relative_path": relative_path,
                "url": f"/viewer/images/{image_id}/file", "annotation_count": len(annotations),
                "category_ids": sorted({annotation["category_id"] for annotation in annotations}), "annotations": annotations,
            })

        db.viewer_images.insert_many(images, ordered=False)
        dataset = {
            "dataset_id": dataset_id, "source": source, "imported_at": now(), "image_count": len(images),
            "annotation_count": sum(image["annotation_count"] for image in images), "skipped_annotations": skipped,
            "categories": [{"category_id": category_ids[label], "name": label} for label in labels],
        }
        db.viewer_datasets.insert_one(dataset)
        staging.replace(final)
        _set_status(db, status="ready", active_dataset=dataset_id, source=source, error=None, download=None)
    except Exception as error:
        db.viewer_images.delete_many({"dataset_id": dataset_id})
        db.viewer_datasets.delete_many({"dataset_id": dataset_id})
        shutil.rmtree(staging, ignore_errors=True)
        _set_failure(db, error)
        raise


def import_default_dataset() -> None:
    db = database()
    archive = upload_dir() / "default-voc.zip"
    try:
        _set_status(db, status="running", source=DEFAULT_DATASET_URL, error=None)
        _download(DEFAULT_DATASET_URL, archive, lambda downloaded, total, percent: _set_status(
            db, status="running", source=DEFAULT_DATASET_URL,
            download={"artifact": "transmission line obj det.zip", "downloaded_bytes": downloaded, "total_bytes": total, "percent": percent},
        ))
        _import_archive(archive, DEFAULT_DATASET_URL)
    except Exception as error:
        _set_failure(db, error)
        raise
    finally:
        archive.unlink(missing_ok=True)


def import_uploaded_dataset(filename: str, source: str) -> None:
    archive = upload_dir() / filename
    if not archive.is_file():
        raise FileNotFoundError("Uploaded dataset archive is unavailable.")
    try:
        _import_archive(archive, source)
    finally:
        archive.unlink(missing_ok=True)


def status(db) -> dict:
    current = db.viewer_imports.find_one({"_id": "active"}) or {"status": "not_imported"}
    dataset_id = current.get("active_dataset")
    dataset = db.viewer_datasets.find_one({"dataset_id": dataset_id}, {"_id": 0}) if dataset_id else None
    return {
        "status": current.get("status", "not_imported"), "job_id": current.get("job_id"), "error": current.get("error"),
        "download": current.get("download"), "source": current.get("source"), "dataset": dataset,
        "categories": (dataset or {}).get("categories", []),
    }


def list_images(db, category_id: int | None, after: str | None, limit: int) -> dict:
    active = db.viewer_imports.find_one({"_id": "active"}) or {}
    dataset_id = active.get("active_dataset")
    if not dataset_id:
        return {"images": [], "next_cursor": None}
    query: dict = {"dataset_id": dataset_id}
    if category_id is not None:
        query["category_ids"] = category_id
    if after:
        query["image_id"] = {"$gt": after}
    rows = list(db.viewer_images.find(query, {"_id": 0, "annotations": 0, "relative_path": 0}).sort("image_id", ASCENDING).limit(limit + 1))
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {"images": rows, "next_cursor": rows[-1]["image_id"] if has_more and rows else None}


def image_details(db, image_id: str) -> dict | None:
    return db.viewer_images.find_one({"image_id": image_id}, {"_id": 0})


def image_path(db, image_id: str) -> Path | None:
    image = db.viewer_images.find_one({"image_id": image_id}, {"dataset_id": 1, "relative_path": 1})
    if image is None:
        return None
    path = (dataset_root() / image["dataset_id"] / image["relative_path"]).resolve()
    root = (dataset_root() / image["dataset_id"]).resolve()
    return path if path.is_relative_to(root) and path.is_file() else None
