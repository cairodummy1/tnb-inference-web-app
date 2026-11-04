# Simple async object detection

A deliberately small, newcomer-friendly object-detection app:

```text
browser → FastAPI → Redis/RQ queue → PyTorch worker → Redis result → browser
```

The web page uploads an image, receives a job ID, polls that ID, and draws the returned bounding boxes on a canvas. It does not need a database: Redis stores the queue and RQ job state/result for this demo.

## Run it

Prerequisites: Docker Compose and roughly 3 GB of free disk/RAM for the first image build and model download.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). The worker downloads the model weights only once and retains them in a Docker volume. Stop with `Ctrl+C`; use `docker compose down` to remove containers while preserving Redis jobs/results and model weights.

### NVIDIA GPU mode

The normal command is CPU-only and works on any supported Docker host. To use an NVIDIA GPU, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), then run:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

The GPU override builds only the worker with PyTorch 2.7.1 and CUDA 12.8, which supports NVIDIA Blackwell GPUs such as the RTX 5090 (`sm_120`), and assigns all host GPUs to it. The application supports `INFERENCE_DEVICE=cpu`, `cuda`, and `auto`; Compose sets `cpu` in the base stack and `cuda` in GPU mode. An explicit CUDA request fails with a clear worker error if Docker cannot access a GPU or the installed wheel does not support the GPU architecture.

## API

| Endpoint | Purpose |
| --- | --- |
| `POST /jobs` | Multipart field `image`; returns `202` plus `{ "job_id", "status" }`. |
| `GET /jobs/{job_id}` | Returns `queued`, `started`, `finished`, or `failed`; includes `result` when finished. |
| `GET /health` | Checks Redis and that an RQ worker is listening on `inference`. |

Results expire after one hour; failed-job records remain for one day. Redis is persisted in a named Docker volume. Uploaded files remain under `data/uploads` for easy inspection and the `cleanup` service deletes files older than 24 hours. Do not use this demo for workloads expected to wait in the queue for over 24 hours.

## Why these choices?

- **FastAPI** handles uploads and makes a small JSON API with little ceremony.
- **Redis + RQ** is the simplest queue fit for a Python-only project: RQ uses ordinary Python functions and gives every queued function a job ID/status. BullMQ would add a Node service just to operate the queue; RabbitMQ is a solid broker but needs more concepts and configuration for this learning app.
- **Torchvision SSDLite320 MobileNetV3-Large (`DEFAULT` / COCO_V1)** is an official public COCO-trained model. It is only 3.4M parameters and 0.58 GFLOPS, making it a practical CPU-first demo. The trade-off is lower COCO box mAP (25.1) than the heavier Faster R-CNN ResNet-50 FPN V2 (46.7 mAP, 43.7M parameters, 280.37 GFLOPS).
- The worker preloads the CPU model once and uses RQ's same-process worker, so PyTorch is not forked after initialization. The first GPU job moves that model to CUDA. This is a sound small-demo trade-off; RQ's same-process worker does not emit periodic job heartbeats, so use the regular RQ worker architecture with a separately managed model server for production workloads that require strict process isolation and heartbeat monitoring.

The model recognizes the 80 standard COCO object categories; it is not a custom detector. Adjust `SCORE_THRESHOLD` in `compose.yaml` to change the 0.5 display cutoff.

## Boundaries of this demo

This is intended for one worker and modest traffic. It has no authentication or upload-malware scanning, and it uses a local bind mount rather than object storage. Before public deployment, add authentication, scanning, structured logging/metrics, object storage, and an observability/alerting strategy. For stronger accuracy on capable hardware, replace the model in `app/worker.py` with Torchvision's `fasterrcnn_resnet50_fpn_v2`.

## Development tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Sources

- [Torchvision object-detection models and official COCO metrics](https://docs.pytorch.org/vision/stable/models.html#object-detection)
- [SSDLite320 MobileNetV3 implementation and COCO weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html)
