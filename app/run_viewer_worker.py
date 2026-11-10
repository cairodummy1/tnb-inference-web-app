"""Run long-lived Pascal VOC viewer imports outside other queues."""
from rq import SimpleWorker

from app.queueing import redis_connection


if __name__ == "__main__":
    SimpleWorker(["viewer-import"], connection=redis_connection()).work()
