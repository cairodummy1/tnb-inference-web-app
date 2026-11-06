"""Run long-lived COCO import jobs outside the image-inference queue."""
from rq import SimpleWorker

from app.queueing import redis_connection


if __name__ == "__main__":
    SimpleWorker(["coco-import"], connection=redis_connection()).work()
