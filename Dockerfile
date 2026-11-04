FROM python:3.11-slim

ARG PYTORCH_WHEEL_CHANNEL=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is required by PyTorch's CPU wheels.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --index-url https://download.pytorch.org/whl/${PYTORCH_WHEEL_CHANNEL} torch==2.6.0 torchvision==0.21.0 \
    && pip install -r requirements.txt

COPY app ./app
