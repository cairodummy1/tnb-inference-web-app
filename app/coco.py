"""COCO 2017 storage, import, and query helpers."""
from __future__ import annotations

import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from pymongo import ASCENDING, MongoClient, UpdateOne

DATASET_ID = "coco2017"
SPLITS = ("train2017", "val2017")
ARCHIVES = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}
BATCH_SIZE = 1_000


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def data_dir() -> Path:
    path = Path(os.environ.get("COCO_DATA_DIR", "data/coco"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def image_root() -> Path:
    path = data_dir() / "images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database():
    client = MongoClient(os.environ.get("MONGODB_URL", "mongodb://localhost:27017"), serverSelectionTimeoutMS=5_000)
    return client[os.environ.get("COCO_DATABASE", "tnb_inference")]


def ensure_indexes(db) -> None:
    db.coco_categories.create_index([("category_id", ASCENDING)], unique=True)
    db.coco_images.create_index([("split", ASCENDING), ("coco_id", ASCENDING)], unique=True)
    db.coco_images.create_index([("category_ids", ASCENDING)])
    db.coco_annotations.create_index([("annotation_id", ASCENDING)], unique=True)
    db.coco_annotations.create_index([("image_key", ASCENDING), ("annotation_id", ASCENDING)])
    db.coco_annotations.create_index([("category_id", ASCENDING)])


def image_key(split: str, coco_id: int) -> str:
    return f"{split}:{coco_id}"


def image_url(key: str) -> str:
    return f"/coco/images/{key}/file"


def _download(url: str, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    temporary.replace(destination)


def _extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(base):
                raise ValueError("COCO archive contains an unsafe path.")
        source.extractall(destination)


def _ensure_downloaded(name: str) -> None:
    root = data_dir()
    if name == "annotations":
        expected = root / "annotations" / "instances_train2017.json"
        destination = root
    else:
        expected = image_root() / name
        destination = image_root()
    if expected.exists():
        return
    archive = root / f"{name}.zip"
    if not archive.exists():
        print(f"Downloading {name}…", flush=True)
        _download(ARCHIVES[name], archive)
    print(f"Extracting {name}…", flush=True)
    _extract(archive, destination)
    archive.unlink(missing_ok=True)


def _bulk(collection, operations: list[UpdateOne]) -> None:
    if operations:
        collection.bulk_write(operations, ordered=False)
        operations.clear()


def _load_instances(split: str) -> dict:
    path = data_dir() / "annotations" / f"instances_{split}.json"
    return json.loads(path.read_text())


def _import_split(db, split: str) -> dict:
    payload = _load_instances(split)
    categories_by_image: dict[int, set[int]] = defaultdict(set)
    annotation_count: Counter[int] = Counter()
    annotation_operations: list[UpdateOne] = []
    for annotation in payload["annotations"]:
        coco_id = int(annotation["image_id"])
        categories_by_image[coco_id].add(int(annotation["category_id"]))
        annotation_count[coco_id] += 1
        document = {
            "annotation_id": int(annotation["id"]), "image_key": image_key(split, coco_id),
            "category_id": int(annotation["category_id"]), "bbox": [float(value) for value in annotation["bbox"]],
            "area": float(annotation["area"]), "iscrowd": bool(annotation["iscrowd"]),
        }
        annotation_operations.append(UpdateOne({"annotation_id": document["annotation_id"]}, {"$set": document}, upsert=True))
        if len(annotation_operations) >= BATCH_SIZE:
            _bulk(db.coco_annotations, annotation_operations)
    _bulk(db.coco_annotations, annotation_operations)

    image_operations: list[UpdateOne] = []
    for image in payload["images"]:
        coco_id = int(image["id"])
        key = image_key(split, coco_id)
        document = {
            "image_key": key, "split": split, "coco_id": coco_id, "filename": image["file_name"],
            "width": int(image["width"]), "height": int(image["height"]), "url": image_url(key),
            "annotation_count": annotation_count[coco_id], "category_ids": sorted(categories_by_image[coco_id]),
        }
        image_operations.append(UpdateOne({"split": split, "coco_id": coco_id}, {"$set": document}, upsert=True))
        if len(image_operations) >= BATCH_SIZE:
            _bulk(db.coco_images, image_operations)
    _bulk(db.coco_images, image_operations)
    summary = {"images": len(payload["images"]), "annotations": len(payload["annotations"])}
    db.coco_imports.update_one({"_id": DATASET_ID}, {"$set": {f"splits.{split}": summary, "updated_at": now()}})
    return summary


def import_coco2017() -> None:
    """Download official archives and upsert COCO train/validation detection data."""
    db = database()
    ensure_indexes(db)
    db.coco_imports.update_one({"_id": DATASET_ID}, {"$set": {"status": "running", "started_at": now(), "error": None}}, upsert=True)
    try:
        _ensure_downloaded("annotations")
        for split in SPLITS:
            _ensure_downloaded(split)
        categories = _load_instances("train2017")["categories"]
        db.coco_categories.bulk_write([
            UpdateOne({"category_id": int(category["id"])}, {"$set": {"category_id": int(category["id"]), "name": category["name"], "supercategory": category["supercategory"]}}, upsert=True)
            for category in categories
        ], ordered=False)
        for split in SPLITS:
            print(f"Importing {split} metadata…", flush=True)
            _import_split(db, split)
        db.coco_imports.update_one({"_id": DATASET_ID}, {"$set": {"status": "ready", "completed_at": now(), "error": None}})
    except Exception as error:
        db.coco_imports.update_one({"_id": DATASET_ID}, {"$set": {"status": "failed", "error": str(error), "updated_at": now()}}, upsert=True)
        raise


def status(db) -> dict:
    record = db.coco_imports.find_one({"_id": DATASET_ID}) or {"status": "not_imported", "splits": {}}
    categories = list(db.coco_categories.find({}, {"_id": 0}).sort("category_id", ASCENDING))
    return {"status": record.get("status", "not_imported"), "splits": record.get("splits", {}), "categories": categories, "error": record.get("error")}


def list_images(db, split: str | None, category_id: int | None, after: str | None, limit: int) -> dict:
    query: dict = {}
    if split:
        query["split"] = split
    if category_id is not None:
        query["category_ids"] = category_id
    if after:
        query["image_key"] = {"$gt": after}
    rows = list(db.coco_images.find(query, {"_id": 0}).sort("image_key", ASCENDING).limit(limit + 1))
    has_more = len(rows) > limit
    rows = rows[:limit]
    return {"images": rows, "next_cursor": rows[-1]["image_key"] if has_more and rows else None}


def image_details(db, key: str) -> dict | None:
    image = db.coco_images.find_one({"image_key": key}, {"_id": 0})
    if image is None:
        return None
    categories = {row["category_id"]: row["name"] for row in db.coco_categories.find({}, {"_id": 0, "category_id": 1, "name": 1})}
    annotations = list(db.coco_annotations.find({"image_key": key}, {"_id": 0}).sort("annotation_id", ASCENDING))
    for annotation in annotations:
        annotation["label"] = categories.get(annotation["category_id"], "unknown")
    image["annotations"] = annotations
    return image
