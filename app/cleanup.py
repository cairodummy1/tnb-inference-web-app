"""Periodically remove old uploaded images from the shared demo data directory."""
from __future__ import annotations

import os
import time
from pathlib import Path


def cleanup_once(upload_dir: Path, retention_seconds: int) -> int:
    cutoff = time.time() - retention_seconds
    removed = 0
    for image_path in upload_dir.iterdir():
        try:
            if image_path.is_file() and image_path.stat().st_mtime < cutoff:
                image_path.unlink()
                removed += 1
        except FileNotFoundError:
            continue
    return removed


def main() -> None:
    upload_dir = Path(os.environ.get("UPLOAD_DIR", "data/uploads"))
    retention_seconds = int(os.environ.get("UPLOAD_RETENTION_SECONDS", "86400"))
    interval_seconds = max(60, int(os.environ.get("CLEANUP_INTERVAL_SECONDS", "3600")))
    upload_dir.mkdir(parents=True, exist_ok=True)
    while True:
        removed = cleanup_once(upload_dir, retention_seconds)
        if removed:
            print(f"Removed {removed} expired upload(s).", flush=True)
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
