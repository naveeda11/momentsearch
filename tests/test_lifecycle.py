"""Document lifecycle regressions: route by kind and clean every artifact."""
from contextlib import contextmanager

import pytest
from psycopg import OperationalError

from src import db
from src.api import videos
from src.ingest import doc_pipeline
from src.ingest.documents import PageRec
from src.rag import embeddings


def test_fifo_document_retry_routes_by_kind(monkeypatch):
    row = {"id": "doc_1", "user_id": "u1", "kind": "paper", "status": "failed"}
    statuses = []
    enqueued = []
    monkeypatch.setattr(videos.db, "get_video", lambda _id: row)
    monkeypatch.setattr(
        videos.db, "set_status",
        lambda source_id, status, **kwargs: statuses.append((source_id, status)))
    monkeypatch.setattr(videos.config, "ENABLE_FAIR_DISPATCH", False)
    monkeypatch.setattr(
        videos.jobs, "enqueue_source",
        lambda source: enqueued.append(source) or "flow-1")

    result = videos.retry("doc_1", "u1")

    assert statuses == [("doc_1", "pending")]
    assert enqueued == [row]
    assert result["flow_run_id"] == "flow-1"


def test_delete_document_removes_pages_checkpoints_and_render(monkeypatch):
    row = {
        "id": "doc_1", "user_id": "u1", "kind": "deck",
        "storage_key": "docs/u1/doc_1.pptx",
    }
    prefixes = []
    keys = []
    monkeypatch.setattr(videos, "is_sample", lambda _id: False)
    monkeypatch.setattr(videos.db, "get_video", lambda _id: row)
    monkeypatch.setattr(videos.db, "delete_video", lambda _id: None)
    monkeypatch.setattr(videos.vector_store, "delete_video", lambda *_args: None)
    monkeypatch.setattr(
        videos.storage, "delete_prefix", lambda prefix: prefixes.append(prefix))
    monkeypatch.setattr(
        videos.storage, "delete_key", lambda key: keys.append(key))

    videos.delete("doc_1", "u1")

    assert videos.storage.page_prefix("u1", "doc_1") in prefixes
    assert videos.storage.frame_prefix("u1", "doc_1") not in prefixes
    assert videos.storage.parsed_key("u1", "doc_1") in keys
    assert videos.storage.doc_render_key("u1", "doc_1") in keys
    assert row["storage_key"] in keys


def test_non_idempotent_db_operation_is_not_retried(monkeypatch):
    calls = 0

    class FakePool:
        @contextmanager
        def connection(self):
            nonlocal calls
            calls += 1
            yield object()

    monkeypatch.setattr(db, "pool", lambda: FakePool())

    with pytest.raises(OperationalError):
        db._run(
            lambda _conn: (_ for _ in ()).throw(OperationalError("lost")),
            retry=False)

    assert calls == 1


def test_caption_checkpoint_written_after_each_success(monkeypatch):
    pages = [
        PageRec(page=1, needs_caption=True),
        PageRec(page=2, needs_caption=True),
    ]
    saves = []
    monkeypatch.setattr(doc_pipeline, "DOC_CAPTION_ENABLED", True)
    monkeypatch.setattr(doc_pipeline.llm, "env_config", lambda: object())
    monkeypatch.setattr(doc_pipeline.llm, "caption_image", lambda *_args: "caption")
    monkeypatch.setattr(doc_pipeline.storage, "get_bytes", lambda _key: b"jpeg")
    monkeypatch.setattr(doc_pipeline.db, "set_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(doc_pipeline.db, "set_progress", lambda *_args: None)
    monkeypatch.setattr(
        doc_pipeline.docs_mod, "save_checkpoint",
        lambda *_args: saves.append(sum(bool(p.caption) for p in pages)))

    result = doc_pipeline.t_doc_enrich.fn(
        "doc_1", "u1", "hash", "deck", pages)

    assert all(p.caption == "caption" for p in result)
    assert len(saves) == 2


def test_remote_text_embeddings_are_memory_bounded(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings.config, "TEXT_EMBED_PROVIDER", "fastembed")
    monkeypatch.setattr(
        embeddings.config, "TEXT_EMBED_SERVICE_URL", "http://text-embed")
    monkeypatch.setattr(embeddings.config, "TEXT_EMBED_BATCH", 2)

    def fake_post(path, payload, **kwargs):
        calls.append((path, list(payload["texts"]), kwargs["service_url"]))
        return {"vectors": [[float(text[-1])] for text in payload["texts"]]}

    monkeypatch.setattr(embeddings, "_post", fake_post)

    result = embeddings.embed_docs(["t1", "t2", "t3", "t4", "t5"])

    assert [len(payload) for _path, payload, _url in calls] == [2, 2, 1]
    assert {url for _path, _payload, url in calls} == {"http://text-embed"}
    assert result.tolist() == [[1.0], [2.0], [3.0], [4.0], [5.0]]
