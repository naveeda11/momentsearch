"""Assignment-contract aliases — thin /admin/* routes over the /api/* handlers.

The Assignment 3 grader scripts (bench.py, eval.py, the eval skill) hardcode
/admin/documents, /admin/sources, and /admin/videos. The repo's real surface
stays /api/*; these aliases delegate to the exact same handler functions so
there is one implementation and two names.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from .documents import DocumentRequest, list_sources, register_document
from .videos import RegisterRequest, register, require_auth, user_id

router = APIRouter(prefix="/admin", tags=["admin-compat"])


@router.post("/documents", status_code=202, dependencies=[Depends(require_auth)])
def admin_documents(req: DocumentRequest, uid: str = Depends(user_id)):
    return register_document(req, uid)


@router.get("/sources")
def admin_sources(uid: str = Depends(user_id)):
    return list_sources(uid)


class AdminVideoRequest(BaseModel):
    url: str
    speaker: str | None = None   # the assignment brief's field — used as title fallback
    title: str | None = None


@router.post("/videos", status_code=202, dependencies=[Depends(require_auth)])
def admin_videos(req: AdminVideoRequest, uid: str = Depends(user_id)):
    out = register(RegisterRequest(url=req.url, title=req.title or req.speaker), uid)
    return {"id": out["video_id"], **out}
