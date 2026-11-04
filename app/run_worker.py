"""Start an RQ worker that keeps the preloaded model in its process."""
from __future__ import annotations

from rq import SimpleWorker

from app.queueing import redis_connection
from app.worker import preload_model, resolve_device


def main() -> None:
    device = resolve_device()
    # SimpleWorker avoids forking a process that has imported/initialized PyTorch.
    # The first job moves this model to CUDA when GPU mode is selected.
    preload_model()
    print(f"Model preloaded on CPU; inference device: {device}", flush=True)
    SimpleWorker(["inference"], connection=redis_connection()).work()


if __name__ == "__main__":
    main()
