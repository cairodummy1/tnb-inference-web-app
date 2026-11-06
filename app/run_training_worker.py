"""Run ANN training jobs separately from image inference jobs."""
from __future__ import annotations

from rq import SimpleWorker

from app.queueing import redis_connection
from app.worker import resolve_device


def main() -> None:
    device = resolve_device()
    print(f"Training worker ready; device: {device}", flush=True)
    SimpleWorker(["training"], connection=redis_connection()).work()


if __name__ == "__main__":
    main()
