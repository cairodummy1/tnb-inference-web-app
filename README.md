# TNB Inference

TNB Inference is a small asynchronous PyTorch demo with a Pascal VOC transmission-line image viewer, image classification, object detection, and numeric tabular-model training.

```text
browser → FastAPI → Redis/RQ queue → worker → database/files → browser
```

## Run it

Prerequisites: Docker Compose and roughly 3 GB of free disk/RAM for the initial image build and model download.

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000) for the transmission-line image viewer, or [http://localhost:8000/inference](http://localhost:8000/inference) for inference and tabular workflows.

DBX's web UI is available at [http://localhost:4224](http://localhost:4224), and Mongo Express at [http://localhost:8081](http://localhost:8081); both are bound to localhost and connect to the `mongodb` Compose service. In DBX, add a MongoDB connection using host `mongodb`, port `27017`, no authentication, and database `tnb_inference`.

## Transmission-line image viewer

The landing-page viewer uses the supplied public Backblaze archive:

```text
https://f005.backblazeb2.com/file/jetson-assets/dataset/transmission%20line%20obj%20det.zip
```

Select **Load transmission-line dataset** in the viewer to queue a download and import. The importer runs independently from inference/training workers, shows progress in the page, stores its files under `data/viewer`, and does not touch legacy datasets under `data/coco`.

The archive is Pascal VOC style: every annotated image has an XML file in an `Annotations/` directory and its image in the sibling `JPEGImages/` directory. XML labels are retained exactly as authored. Invalid image/XML pairs are skipped and reported after import.

### Use your own VOC dataset

Start the stack, then upload a local Pascal VOC ZIP to activate it in the viewer:

```bash
python scripts/upload_voc_dataset.py /path/to/my-dataset.zip
```

For a non-default local address, supply `--api-url`, for example:

```bash
python scripts/upload_voc_dataset.py ./inspection-data.zip --api-url http://localhost:8000
```

The script uploads to the running local API, queues the import, waits for completion, and only switches the active viewer dataset after a successful import. A ZIP must contain matching `Annotations/*.xml` and `JPEGImages/*` paths; archive paths are validated before extraction.

## API

| Endpoint | Purpose |
| --- | --- |
| `GET /viewer/status` | Active viewer dataset, import state, categories, and download progress. |
| `POST /viewer/import-jobs` | Queue the default public transmission-line dataset import. |
| `POST /viewer/datasets` | Upload and queue a custom Pascal VOC ZIP. |
| `GET /viewer/images` | Cursor-paginated active images; supports `category_id`, `cursor`, and `limit`. |
| `GET /viewer/images/{image_id}` | Image metadata and Pascal VOC annotations. |
| `GET /viewer/images/{image_id}/file` | Stream the imported image file. |
| `POST /jobs` | Multipart field `image`; queues object detection. |
| `POST /classification/jobs` | Multipart image; queues ResNet-50 classification. |
| `GET /jobs/{job_id}` | Returns job state and result when finished. |
| `POST /tabular/datasets` | Upload a CSV/XLSX/XLS file and return its preview. |
| `POST /tabular/training-jobs` | Queue ANN training from an uploaded table. |
| `GET /health` | Checks Redis and that an inference worker is available. |

## NVIDIA GPU mode

The normal command is CPU-only. For an NVIDIA GPU, install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), configure it as Docker's `nvidia` runtime, and restart Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then run:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up --build
```

### Jetson Orin

Do not use `compose.gpu.yaml` on an Orin: its standard CUDA PyTorch wheel may not include the Orin GPU's `sm_87` architecture. Use the Jetson-specific PyTorch/torchvision image instead:

```bash
docker compose -f compose.yaml -f compose.jetson.yaml up --build
```

The Jetson overlay defaults to `dustynv/l4t-pytorch:r36.4.0` (ARM64). If your installed L4T/JetPack release requires another compatible image, set `JETSON_PYTORCH_IMAGE` to that image tag before building. At startup, each worker checks CUDA on the actual GPU and runs `torchvision` NMS on CUDA tensors before joining its queue; an incompatible image fails clearly rather than registering as ready.

## Development tests

```bash
pip install -r requirements-dev.txt
pytest
```
