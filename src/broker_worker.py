"""Redis Streams consumer — the self-hosted replacement for Prefect serving.

    QUEUE_BACKEND=redis python -m src.broker_worker

Loop: XREADGROUP new messages -> execute the matching ingest flow in-process
-> XACK only after the flow returns (ack-after-upsert). Every tick also
XAUTOCLAIMs messages another consumer left pending longer than the visibility
timeout, and dead-letters anything delivered more than BROKER_MAX_DELIVERIES
times. Scale by running more replicas — the consumer group splits the stream.
"""
from __future__ import annotations

import os
import socket
import time

from . import broker, db
from .broker import GROUP, MAX_DELIVERIES, STREAM, VISIBILITY_S

CONSUMER = f"{socket.gethostname()}-{os.getpid()}"


def _execute(fields: dict) -> None:
    """Run the right ingest flow for one message, synchronously, in-process.
    Prefect @flow objects called directly just execute locally — same code
    path as the managed queue, different delivery mechanism."""
    from .ingest.doc_pipeline import ingest_document
    from .ingest.pipeline import ingest_video

    kind = fields.get("kind") or "video"
    if kind == "video":
        ingest_video(fields["id"], fields["user_id"])
    else:
        ingest_document(fields["id"], fields["user_id"], kind)


def _handle(msg_id: str, fields: dict) -> None:
    try:
        _execute(fields)
    except Exception as exc:  # noqa: BLE001 — flow already marked the row failed
        print(f"[broker-worker] {fields.get('id')} failed: {type(exc).__name__}: {exc}")
    finally:
        # ACK regardless: a *crash* (the case redelivery exists for) never
        # reaches this line; a handled Python failure has already written the
        # row's terminal status, so redelivering it would be wasted work.
        broker.client().xack(STREAM, GROUP, msg_id)


def _reclaim() -> None:
    """Visibility timeout: adopt messages a dead consumer left pending; DLQ
    anything that keeps killing its consumer."""
    c = broker.client()
    next_id = "0-0"
    while True:
        next_id, messages, _ = c.xautoclaim(
            STREAM, GROUP, CONSUMER, min_idle_time=VISIBILITY_S * 1000,
            start_id=next_id, count=10)
        if not messages:
            break
        pending = {p["message_id"]: p for p in
                   c.xpending_range(STREAM, GROUP, min="-", max="+", count=200)}
        for msg_id, fields in messages:
            times = pending.get(msg_id, {}).get("times_delivered", 1)
            if times > MAX_DELIVERIES:
                print(f"[broker-worker] DLQ {fields.get('id')} "
                      f"(delivered {times}x — poison)")
                db.set_status(fields["id"], "failed",
                              error=f"dead-letter: {times} deliveries (broker)")
                broker.dead_letter(msg_id, fields, f"{times} deliveries")
                continue
            print(f"[broker-worker] reclaimed {fields.get('id')} "
                  f"(idle > {VISIBILITY_S}s, delivery {times})")
            _handle(msg_id, fields)
        if next_id == "0-0":
            break


def main() -> None:
    db.init_schema()
    from .rag import vector_store
    vector_store.ensure_collection()
    vector_store.ensure_text_collection()
    broker.ensure_group()
    # Same WFQ fairness as the Prefect path: the dispatcher claims pending rows
    # round-robin across tenants and (with QUEUE_BACKEND=redis) XADDs them here.
    from . import dispatcher
    dispatcher.start_in_background()
    print(f"[broker-worker] consuming {STREAM} as {GROUP}/{CONSUMER} "
          f"(visibility {VISIBILITY_S}s, max deliveries {MAX_DELIVERIES})")
    last_reclaim = 0.0
    while True:
        try:
            if time.time() - last_reclaim > 30:
                _reclaim()
                last_reclaim = time.time()
            resp = broker.client().xreadgroup(
                GROUP, CONSUMER, {STREAM: ">"}, count=1, block=5000)
            for _stream, messages in resp or []:
                for msg_id, fields in messages:
                    _handle(msg_id, fields)
        except KeyboardInterrupt:
            break
        except Exception as exc:  # noqa: BLE001 — never die; Redis blips heal
            print(f"[broker-worker] loop error: {type(exc).__name__}: {exc} — 5s")
            time.sleep(5)


if __name__ == "__main__":
    main()
