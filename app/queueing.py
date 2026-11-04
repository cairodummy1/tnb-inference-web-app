"""Shared Redis/RQ helpers for the API and Compose health checks."""
from __future__ import annotations

import os

from redis import Redis
from rq import Worker


def redis_connection() -> Redis:
    return Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))


def has_inference_worker(connection: Redis) -> bool:
    """Return whether a live RQ worker is listening on the inference queue."""
    return any("inference" in worker.queue_names() for worker in Worker.all(connection))
