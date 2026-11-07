# TNB Inference

TNB Inference is a deliberately small, newcomer-friendly asynchronous PyTorch demo with COCO object detection, ImageNet image classification, a file-trained ANN for numeric tabular prediction, and a MongoDB-backed COCO 2017 dataset browser.

```text
browser → FastAPI → Redis/RQ queue → PyTorch worker → Redis result → browser
```

The web page uploads images or tabular datasets, receives a job ID, and polls it for results. Redis stores RQ job state/results; the latest successful ANN model and its source-file metadata are stored in the shared data volume.

## Run it

Prerequisites: Docker Compose and roughly 3 GB of free disk/RAM for the first image build and model download.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the COCO Image Viewer, or [http://localhost:8000/inference](http://localhost:8000/inference) for object detection, CNN classification, and ANN workflows. The worker downloads model weights only once and retains them in a Docker volume. Stop with `Ctrl+C`; use `docker compose down` to remove containers while preserving Redis jobs/results and model weights.

### Import the complete COCO 2017 dataset

The dataset browser imports official COCO 2017 train and validation object-detection data only. It downloads approximately 123k images to `data/coco/images`, saves categories/images/annotations in MongoDB, and serves JPEGs through FastAPI. Reserve roughly 50 GB of local disk space for sequential archive download and extraction.

Start the normal stack first, then open the **Image Viewer** landing page and select **Start full COCO import**. FastAPI queues the long-running job for the dedicated RQ COCO worker, so the API, image inference, and ANN training remain available. The import is resumable and safe to retry after interruption; the viewer shows queue/download progress and completion state.

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
| `POST /classification/jobs` | Multipart image; queues ResNet-50 ImageNet classification. |
| `GET /jobs/{job_id}` | Returns `queued`, `started`, `finished`, or `failed`; includes `result` when finished. |
| `POST /tabular/datasets` | Upload a CSV/XLSX/XLS file and return its dataset ID and preview. |
| `GET /tabular/datasets/{dataset_id}/preview` | Preview a CSV or selected Excel worksheet. |
| `POST /tabular/training-jobs` | Queue ANN training with dataset ID, worksheet, target, and task. |
| `GET /tabular/model` | Return metadata for the latest successful ANN model. |
| `POST /tabular/predictions` | Predict from the latest ANN with an exact feature-value schema. |
| `GET /coco/status` | Return COCO import state, split counts, and categories. |
| `POST /coco/import-jobs` | Queue the full COCO 2017 download and MongoDB import. |
| `GET /coco/images` | Cursor-paginated COCO image metadata with optional split/category filters. |
| `GET /coco/images/{image_key}` | Return one image and its ground-truth COCO boxes/labels. |
| `GET /coco/images/{image_key}/file` | Stream the server-stored COCO JPEG. |
| `GET /health` | Checks Redis and that an RQ worker is listening on `inference`. |

Results expire after one hour; failed-job records remain for one day. Redis is persisted in a named Docker volume. Temporary images and datasets remain under `data/uploads` and `data/tabular/datasets` for easy inspection and the `cleanup` service deletes them after 24 hours. The latest successful ANN run, saved model state, metadata, and source-file copy remain under `data/tabular/models`.

## Why these choices?

- **FastAPI** handles uploads and makes a small JSON API with little ceremony.
- **Redis + RQ** is the simplest queue fit for a Python-only project: RQ uses ordinary Python functions and gives every queued function a job ID/status. BullMQ would add a Node service just to operate the queue; RabbitMQ is a solid broker but needs more concepts and configuration for this learning app.
- **Torchvision SSDLite320 MobileNetV3-Large (`DEFAULT` / COCO_V1)** is an official public COCO-trained model. It is only 3.4M parameters and 0.58 GFLOPS, making it a practical CPU-first demo. The trade-off is lower COCO box mAP (25.1) than the heavier Faster R-CNN ResNet-50 FPN V2 (46.7 mAP, 43.7M parameters, 280.37 GFLOPS).
- **Torchvision ResNet-50 (`DEFAULT` / ImageNet-1K)** returns the top five predictions from ImageNet's 1,000 general categories. It is not a custom TNB classifier; fine-tuning is required for domain-specific labels.
- **Custom ANN** accepts numeric feature columns only. The browser previews the uploaded CSV/XLSX dataset, lets the user choose one target column and classification or regression, then trains a 128→64 PyTorch MLP. The latest successful model stores its feature order, scaling data, target/classes, metrics, source file, and training settings so the prediction form always matches its schema.
- **COCO 2017 Image Viewer** stores categories, image metadata, and detection annotations in MongoDB. It filters full images by class or all objects; opening an image provides an optional bounding-box and label overlay. JPEGs remain in the server’s `data/coco/images` directory and MongoDB stores a FastAPI image URL rather than binary image data.
- The image worker preloads its public CPU models once and uses RQ's same-process worker, so PyTorch is not forked after initialization. A separate training worker prevents long ANN training from blocking image jobs. GPU Compose grants both workers NVIDIA GPU access.

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
- [Torchvision classification models and ResNet-50 ImageNet weights](https://docs.pytorch.org/vision/main/models)
- [Official COCO 2017 detection dataset](https://cocodataset.org/dataset/detection-2017.htm)
