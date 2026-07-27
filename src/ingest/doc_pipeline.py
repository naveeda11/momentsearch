"""Per-document ingest pipeline — the Prefect flow for papers and decks.

pending -> fetching -> parsing -> enriching -> embedding -> indexed | skipped | failed

Mirrors the video flow (pipeline.py) stage for stage:
  1. fetch    acquire the PDF/PPTX into scratch (URL download or the docs/
              storage checkpoint), hash it, skip duplicates
  2. parse    pymupdf / python-pptx -> per-page text + JPEG renders uploaded to
              pages/{user}/{doc}/NNNNNN.jpg; checkpoint JSON to parsed/
  3. enrich   vision-LLM captions for image-only pages (figures, charts,
              scans) — separate task because it is network-bound and flaky;
              a caption retry must never re-parse. Best-effort like the video
              transcript stage: it can never fail the flow.
  4. index    chunk (page-aware / slide-aware) -> embed_docs -> upsert into the
              SAME text collection as video transcripts, `kind` + locator in
              every payload. `indexed` is written ONLY after the last wait=True
              upsert — the crash-safe ordering the resilience gate checks.

Crash resume: fetch finds the raw doc in storage, parse loads the parsed/
checkpoint (same source_hash), enrich skips already-captioned pages, index
overwrites deterministic point IDs. A killed run redoes almost nothing.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from prefect import flow, task

from .. import db, llm, storage
from ..config import (
    DOC_CAPTION_ENABLED,
    TEXT_EMBED_VERSION,
)
from ..rag import vector_store
from ..rag.embeddings import embed_docs
from . import documents as docs_mod
from .documents import PageRec, UnsupportedDocument

_UPLOAD_POOL = 8
_CAPTION_POOL = 4     # concurrent vision-LLM caption calls
_EMBED_BATCH = 256    # text chunks per embed/upsert call


@task(name="doc-fetch", retries=2, retry_delay_seconds=[30, 120])
def t_doc_fetch(doc_id: str, user_id: str) -> tuple[str, str]:
    """Document -> worker scratch. Returns (path, source_hash); ("", "") when
    the content duplicates an already-indexed source (row marked 'skipped')."""
    db.set_status(doc_id, "fetching")
    row = db.get_video(doc_id)
    if row is None:
        # Row deleted between scheduling and execution — permanent, and
        # retrying would hold a worker slot for 30s+120s doing nothing.
        print(f"[fetch] {doc_id}: manifest row deleted — nothing to ingest")
        return "", ""
    try:
        path, source_hash, storage_key = docs_mod.fetch_document(row, user_id, doc_id)
    except UnsupportedDocument as exc:
        # Permanent — a URI that will never be a PDF/PPTX must not burn task
        # retries (30s+120s each) nor a Prefect "Failed" run. Mark the row
        # failed (the business source of truth) and end the flow cleanly.
        db.set_status(doc_id, "failed", error=f"unsupported: {exc}")
        print(f"[fetch] {doc_id}: unsupported document ({exc}) — failed, no retry")
        return "", ""
    db.set_status(doc_id, "fetching",
                  source_hash=source_hash, storage_key=storage_key,
                  title=row.get("title") or Path(path).stem)

    dup = db.find_duplicate(user_id, source_hash, exclude_id=doc_id)
    if dup:
        Path(path).unlink(missing_ok=True)
        db.set_status(doc_id, "skipped", error=f"duplicate of {dup['id']}")
        return "", ""
    return str(path), source_hash


@task(name="doc-parse", retries=1, retry_delay_seconds=30)
def t_doc_parse(doc_id: str, user_id: str, path: str, source_hash: str) -> list[PageRec]:
    """Parse to per-page text + upload page renders. Checkpoint short-circuit:
    a prior run's parse of the same content is loaded, not redone."""
    db.set_status(doc_id, "parsing", progress=0.0)

    cached = docs_mod.load_checkpoint(user_id, doc_id, source_hash)
    if cached is not None:
        print(f"[parse] {doc_id}: resume from checkpoint ({len(cached)} pages, renders kept)")
        db.set_status(doc_id, "parsing", page_count=len(cached), progress=1.0)
        return cached

    pages, renders = docs_mod.parse_document(Path(path))
    if not pages:
        raise RuntimeError("document parsed to zero pages/slides")

    # Idempotent re-run: clear renders a previous attempt half-uploaded.
    storage.delete_prefix(storage.page_prefix(user_id, doc_id))
    with_render = [(p.page, r) for p, r in zip(pages, renders) if r]
    done = 0

    def _put(item: tuple[int, bytes]) -> None:
        nonlocal done
        page_no, jpeg = item
        storage.put_bytes(storage.page_key(user_id, doc_id, page_no), jpeg, "image/jpeg")
        done += 1
        if done % 10 == 0:
            db.set_progress(doc_id, done / max(len(with_render), 1))

    with ThreadPoolExecutor(max_workers=_UPLOAD_POOL) as ex:
        list(ex.map(_put, with_render))

    docs_mod.save_checkpoint(user_id, doc_id, source_hash, pages)
    n_img = sum(1 for p in pages if p.needs_caption)
    print(f"[parse] {doc_id}: {len(pages)} pages, {len(with_render)} renders, "
          f"{n_img} image-only")
    db.set_status(doc_id, "parsing", page_count=len(pages), progress=1.0)
    return pages


@task(name="doc-enrich", retries=2, retry_delay_seconds=30)
def t_doc_enrich(doc_id: str, user_id: str, source_hash: str, kind: str,
                 pages: list[PageRec]) -> list[PageRec]:
    """Caption image-only pages with the server vision LLM. Best-effort: no
    LLM configured or every caption failing still leaves text-only retrieval
    working — this stage never fails the flow. The checkpoint is updated after
    each caption, so a crash loses at most one call."""
    db.set_status(doc_id, "enriching", progress=0.0)
    cfg = llm.env_config()
    todo = [p for p in pages if p.needs_caption and not p.caption]
    if not DOC_CAPTION_ENABLED or cfg is None or not todo:
        if todo:
            print(f"[enrich] {doc_id}: no vision LLM configured — "
                  f"{len(todo)} image-only pages stay uncaptioned")
        return pages

    what = "slide" if kind == "deck" else "page of a research paper"
    done = 0

    def _caption(p: PageRec) -> None:
        nonlocal done
        try:
            jpeg = storage.get_bytes(storage.page_key(user_id, doc_id, p.page))
            p.caption = llm.caption_image(jpeg, cfg, what)
        except Exception as exc:
            print(f"[enrich] {doc_id} p.{p.page}: caption failed "
                  f"({type(exc).__name__}: {exc}) — continuing")
        done += 1
        db.set_progress(doc_id, done / len(todo))

    try:
        with ThreadPoolExecutor(max_workers=_CAPTION_POOL) as ex:
            list(ex.map(_caption, todo))
        docs_mod.save_checkpoint(user_id, doc_id, source_hash, pages)
        ok = sum(1 for p in todo if p.caption)
        print(f"[enrich] {doc_id}: captioned {ok}/{len(todo)} image-only pages")
    except Exception as exc:
        print(f"[enrich] {doc_id}: enrich crashed ({type(exc).__name__}: {exc}) "
              f"— continuing text-only")
    return pages


@task(name="doc-index", retries=2, retry_delay_seconds=60)
def t_doc_index(doc_id: str, user_id: str, kind: str, pages: list[PageRec]) -> int:
    """Chunk -> embed -> upsert into the shared text collection. The `indexed`
    status is written strictly AFTER the last successful wait=True upsert."""
    db.set_status(doc_id, "embedding", progress=0.0)
    chunks = docs_mod.chunk_pages(pages, kind)
    if not chunks:
        raise RuntimeError("no text extracted — nothing to index")

    vector_store.ensure_text_collection()
    vector_store.delete_video(user_id, doc_id)  # drop stale points from prior runs

    locator = "slide" if kind == "deck" else "page"
    total = 0
    for start in range(0, len(chunks), _EMBED_BATCH):
        batch = chunks[start:start + _EMBED_BATCH]
        vecs = embed_docs([c["text"] for c in batch])
        payloads = [{"user_id": user_id, "video_id": doc_id, "kind": kind,
                     "modality": "text", "text": c["text"], locator: c[locator],
                     "embed_version": TEXT_EMBED_VERSION} for c in batch]
        vector_store.upsert_chunks_at(user_id, doc_id, vecs, payloads, offset=start)
        total += len(batch)
        db.set_progress(doc_id, total / len(chunks))

    db.set_status(doc_id, "indexed", chunk_count=total, progress=1.0)
    return total


@flow(name="ms-ingest-document", log_prints=True, timeout_seconds=3600)
def ingest_document(doc_id: str, user_id: str, kind: str = "paper") -> dict:
    attempt = db.bump_attempts(doc_id)
    path: str | None = None
    try:
        path, source_hash = t_doc_fetch(doc_id, user_id)
        if not path:  # duplicate — already marked 'skipped'
            print(f"[ingest-doc] {doc_id} skipped (duplicate content)")
            return {"id": doc_id, "skipped": True}
        pages = t_doc_parse(doc_id, user_id, path, source_hash)
        pages = t_doc_enrich(doc_id, user_id, source_hash, kind, pages)
        n = t_doc_index(doc_id, user_id, kind, pages)
        print(f"[ingest-doc] {doc_id} indexed: {len(pages)} pages -> {n} chunks "
              f"(attempt {attempt})")
        return {"id": doc_id, "pages": len(pages), "chunks": n}
    except Exception as exc:
        db.set_status(doc_id, "failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if path:
            Path(path).unlink(missing_ok=True)
