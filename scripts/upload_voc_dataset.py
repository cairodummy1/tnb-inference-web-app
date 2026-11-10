#!/usr/bin/env python3
"""Upload a local Pascal VOC ZIP to the running image viewer."""
from __future__ import annotations

import argparse
import http.client
import json
import mimetypes
import sys
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request(url: str, method: str = "GET", body: bytes | None = None, content_type: str | None = None) -> dict:
    headers = {"Content-Type": content_type} if content_type else {}
    try:
        with urlopen(Request(url, data=body, headers=headers, method=method), timeout=60) as response:
            return json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"Server returned {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach the image viewer: {error.reason}") from error


def upload_file(api_url: str, path: Path) -> dict:
    boundary = f"----voc-upload-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(path.name)[0] or "application/zip"
    head = f"--{boundary}\r\nContent-Disposition: form-data; name=\"dataset\"; filename=\"{path.name}\"\r\nContent-Type: {content_type}\r\n\r\n".encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    from urllib.parse import urlsplit
    target = urlsplit(api_url)
    connection = http.client.HTTPSConnection(target.netloc, timeout=60) if target.scheme == "https" else http.client.HTTPConnection(target.netloc, timeout=60)
    try:
        connection.putrequest("POST", f"{target.path.rstrip('/')}/viewer/datasets")
        connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
        connection.putheader("Content-Length", str(len(head) + path.stat().st_size + len(tail)))
        connection.endheaders()
        connection.send(head)
        with path.open("rb") as archive:
            while chunk := archive.read(1024 * 1024):
                connection.send(chunk)
        connection.send(tail)
        response = connection.getresponse()
        payload = response.read().decode("utf-8", "replace")
        if response.status >= 400:
            raise RuntimeError(f"Server returned {response.status}: {payload}")
        return json.loads(payload)
    except OSError as error:
        raise RuntimeError(f"Could not reach the image viewer: {error}") from error
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload and activate a Pascal VOC dataset ZIP.")
    parser.add_argument("archive", type=Path, help="Local .zip containing Annotations/ and JPEGImages/ directories")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Running FastAPI base URL")
    args = parser.parse_args()
    if args.archive.suffix.lower() != ".zip" or not args.archive.is_file():
        parser.error("archive must be an existing .zip file")
    api = args.api_url.rstrip("/")
    job = upload_file(api, args.archive)
    print(f"Import queued: {job['job_id']}")
    while True:
        status = request(f"{api}/viewer/status")
        print(f"{status['status']}", end="\r", flush=True)
        if status["status"] == "ready":
            dataset = status["dataset"] or {}
            print(f"\nActive dataset: {dataset.get('image_count', 0)} images, {dataset.get('annotation_count', 0)} annotations")
            return 0
        if status["status"] == "failed":
            print(f"\nImport failed: {status.get('error')}", file=sys.stderr)
            return 1
        time.sleep(2)


if __name__ == "__main__":
    raise SystemExit(main())
