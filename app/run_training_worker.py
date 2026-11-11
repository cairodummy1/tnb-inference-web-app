"""Run ANN training jobs separately from image inference jobs."""
from __future__ import annotations

import torch
from rq import SimpleWorker

from app.queueing import redis_connection
from app.worker import resolve_device, validate_cuda_runtime


def main() -> None:
    device = resolve_device()
    validate_cuda_runtime(device)
    if device.type == "cuda":
        capability = torch.cuda.get_device_capability(device)
        print(
            f"Training worker ready; device: {device} "
            f"(sm_{capability[0]}{capability[1]})",
            flush=True,
        )
    else:
        print(f"Training worker ready; device: {device}", flush=True)
    SimpleWorker(["training"], connection=redis_connection()).work()


if __name__ == "__main__":
    main()
