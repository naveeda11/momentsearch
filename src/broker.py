"""Self-hosted work queue — Redis Streams behind the same enqueue interface.

The stretch goal's point: understand what the managed queue (Prefect Cloud)
was giving us by building the guarantees ourselves.

    QUEUE_BACKEND=prefect   (default) API/dispatcher schedule Prefect runs
    QUEUE_BACKEND=redis     producers XADD to a stream; broker_worker.py
                            consumers execute the same ingest flows

Semantics implemented here:
  * At-least-once delivery — a consumer group tracks every message until it is
    XACKed; a message is only ACKed AFTER the ingest flow finishes and the
    row's terminal status is committed (ack-after-upsert). A consumer that
    dies mid-flow leaves the message pending.
  * Visibility timeout — XAUTOCLAIM hands messages pending longer than
    BROKER_VISIBILITY_S to a live consumer (the crashed-worker recovery).
  * Dead-letter queue — a message delivered more than BROKER_MAX_DELIVERIES
    times moves to the ms:ingest:dlq stream and the row is marked failed, so
    a poison source can't loop forever.

Idempotency comes from the pipeline itself (deterministic point IDs, storage
checkpoints), which is exactly what makes at-least-once delivery safe.
"""
from __future__ import annotations

import os

STREAM = os.getenv("BROKER_STREAM", "ms:ingest")
DLQ_STREAM = f"{STREAM}:dlq"
GROUP = os.getenv("BROKER_GROUP", "ms-workers")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
VISIBILITY_S = int(os.getenv("BROKER_VISIBILITY_S", "600"))
MAX_DELIVERIES = int(os.getenv("BROKER_MAX_DELIVERIES", "3"))

_client = None


def client():
    global _client
    if _client is None:
        import redis
        _client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    return _client


def ensure_group() -> None:
    """Create the stream + consumer group (idempotent)."""
    import redis
    try:
        client().xgroup_create(STREAM, GROUP, id="0", mkstream=True)
    except redis.ResponseError as exc:  # BUSYGROUP = already exists
        if "BUSYGROUP" not in str(exc):
            raise


def enqueue(row: dict) -> str:
    """Producer side (API / dispatcher): one message per source."""
    return client().xadd(STREAM, {"id": row["id"], "user_id": row["user_id"],
                                  "kind": row.get("kind") or "video"})


def dead_letter(msg_id: str, fields: dict, reason: str) -> None:
    """Move a poison message to the DLQ and ACK it off the main stream."""
    client().xadd(DLQ_STREAM, {**fields, "reason": reason, "original_id": msg_id})
    client().xack(STREAM, GROUP, msg_id)
