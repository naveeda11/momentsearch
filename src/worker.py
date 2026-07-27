"""Ingest worker entrypoint — serves the Prefect flow.

    python -m src.worker

flow.serve() registers the "ms-ingest-video/ingest" deployment in Prefect Cloud
(idempotent) and long-polls for scheduled runs — outbound HTTPS only, no
ports. Scale horizontally by running more replicas of this process; each
executes up to WORKER_CONCURRENCY runs at once.

Sample seeding is NOT done here — it's a one-shot startup gate (seed.py /
src/seeding.py) that the whole stack waits on, so the app never serves a
half-indexed corpus. This worker only handles user uploads + YouTube adds.

Embedding goes to the warm CLIP service when CLIP_SERVICE_URL is set
(docker-compose default); unset, each run loads the model in-process.
"""
import os
import time

from prefect import serve
from prefect.deployments.runner import EntrypointType

from .db import init_schema
from .ingest.doc_pipeline import ingest_document
from .ingest.pipeline import ingest_video


def main():
    init_schema()  # make sure migrations ran before consuming runs
    from .rag import vector_store
    vector_store.ensure_collection()  # up front, not mid-first-ingest
    vector_store.ensure_text_collection()  # docs index here even without transcripts
    # Fair scheduler (WFQ): admits pending videos round-robin across users so
    # one bulk uploader can't starve everyone else (src/dispatcher.py).
    from . import dispatcher
    dispatcher.start_in_background()
    limit = int(os.getenv("WORKER_CONCURRENCY", "2"))
    # serve() talks to Prefect Cloud on startup; a transient outage (e.g. a 503)
    # used to crash the worker permanently and stop the machine. Self-heal:
    # retry forever so a blip pauses ingest instead of killing the worker.
    while True:
        try:
            print(f"[worker] serving 'ms-ingest-video/ingest' + "
                  f"'ms-ingest-document/ingest-doc' (concurrency {limit})")
            # MODULE_PATH: the runner re-imports each flow as src.ingest.<mod>
            # (a real package import), so the modules' relative imports work.
            # The default FILE_PATH executes the file as a loose script, where
            # `from .. import db` dies with "beyond top-level package".
            serve(ingest_video.to_deployment(
                      name="ingest", entrypoint_type=EntrypointType.MODULE_PATH),
                  ingest_document.to_deployment(
                      name="ingest-doc", entrypoint_type=EntrypointType.MODULE_PATH),
                  limit=limit)
            break  # clean shutdown
        except KeyboardInterrupt:
            break
        except Exception as exc:
            print(f"[worker] serve crashed: {type(exc).__name__}: {exc} — retrying in 15s")
            time.sleep(15)


if __name__ == "__main__":
    main()
