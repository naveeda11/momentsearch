"""Document registration API — papers (PDF) and decks (PDF/PPTX).

Same async contract as /api/videos: the request path only validates shape,
writes a `pending` manifest row, and returns 202. All fetching/parsing/
embedding happens on a queue worker — a 60-page paper can never make this
endpoint (or a concurrent search) slow.

Also serves GET /api/sources: the unified status view across every kind
(video + paper + deck), which the benchmark polls for throughput and the
resilience gate polls for terminal states.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import config, db, jobs
from .videos import require_auth, user_id

router = APIRouter(prefix="/api", tags=["documents"])

_KINDS = ("paper", "deck")


class DocumentRequest(BaseModel):
    uri: str | None = None    # http(s) URL or storage://<key>
    key: str | None = None    # object-storage key (already uploaded)
    kind: str
    title: str | None = None


def register_document(req: DocumentRequest, uid: str) -> dict:
    if req.kind not in _KINDS:
        raise HTTPException(400, f"kind must be one of {_KINDS}.")
    uri = (req.uri or "").strip()
    key = (req.key or "").strip()
    if uri.startswith("storage://"):
        key, uri = uri[len("storage://"):], ""
    if not uri and not key:
        raise HTTPException(400, "Provide a uri (http(s) or storage://) or a key.")
    if uri and not uri.startswith(("http://", "https://")):
        raise HTTPException(400, "uri must be http(s) or storage://<key>.")
    if key and not key.startswith(f"{config.DOC_KEY_PREFIX}{uid}/"):
        raise HTTPException(403, "Key does not belong to this user.")

    doc_id = f"doc_{uuid.uuid4().hex[:10]}"
    row = db.upsert_pending({"id": doc_id, "user_id": uid, "source": "document",
                             "kind": req.kind, "url": uri or None,
                             "storage_key": key or None, "source_hash": None,
                             "title": req.title})
    # Fair dispatch (default): leave it pending — the WFQ dispatcher admits it
    # and routes by kind (jobs.enqueue_source). FIFO mode: enqueue right away.
    out = {"id": row["id"], "video_id": row["id"], "status": "pending",
           "kind": req.kind}
    if not config.ENABLE_FAIR_DISPATCH:
        out["flow_run_id"] = jobs.enqueue_document(row["id"], uid, req.kind)
    return out


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def create_document(req: DocumentRequest, uid: str = Depends(user_id)):
    return register_document(req, uid)


def list_sources(uid: str) -> dict:
    sources = []
    for r in db.list_videos(uid):
        sources.append({
            "id": r["id"],
            "kind": r.get("kind") or "video",
            "status": r["status"],
            "title": r.get("title"),
            "pct": round((r.get("progress") or 0.0) * 100),
            "chunks": r.get("chunk_count"),
            "pages": r.get("page_count"),
            "frames": r.get("frame_count"),
            "error": r.get("error"),
            "attempts": r.get("attempts"),
            "created_at": r.get("created_at"),
            "updated_at": r.get("updated_at"),
        })
    return {"sources": sources}


@router.get("/sources")
def get_sources(uid: str = Depends(user_id)):
    return list_sources(uid)
