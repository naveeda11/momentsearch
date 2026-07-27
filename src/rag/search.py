"""Read path: question -> retrieve -> gate -> cited answer (or honest abstain).

Retrieval is milliseconds; the multimodal LLM call is seconds and dominates
cost. So the shape is a confidence funnel: fetch KNN_K candidates, collapse
temporal near-duplicates, trim to TOP_K, and — Gate 1 — if even the best
score is below CONFIDENCE_THRESHOLD, abstain WITHOUT calling the LLM. That
one free check kills most hallucination risk. Generated answers get their
[n] citations validated; invented references are stripped.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .. import config, db, llm, storage
from ..config import (BRANCH_TOP_K, CONFIDENCE_THRESHOLD, CROSS_MODAL_BOOST,
                      FUSION_WINDOW_S, RRF_K, TEXT_CONFIDENCE_THRESHOLD, TOP_K)
from . import vector_store
from .embeddings import embed_query, embed_text

ABSTAIN = ("I couldn't find that in your sources — nothing indexed looks "
           "related to the question (no video moment, paper page, or slide).")


def _seconds(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


_DECK_INTENT = re.compile(r"\b(slides?|deck|keynote|presentation)\b", re.I)
_PAPER_INTENT = re.compile(r"\b(papers?|survey|page|pdf|article|study)\b", re.I)


def _kind_intent(question: str) -> list[str] | None:
    """Source kinds the question names explicitly, if any."""
    kinds = []
    if _DECK_INTENT.search(question):
        kinds.append("deck")
    if _PAPER_INTENT.search(question):
        kinds.append("paper")
    return kinds or None


def _locator_key(h: dict) -> tuple[str, int] | None:
    """A document hit's bucket key: pages/slides have no time axis — every doc
    chunk carries t=0, so time-windowing would collapse a whole paper into one
    citation. Docs bucket by exact (page|slide) number instead; video keeps the
    time window. None = time-based (video)."""
    if h.get("page") is not None:
        return ("page", int(h["page"]))
    if h.get("slide") is not None:
        return ("slide", int(h["slide"]))
    return None


def _fuse(visual_hits: list[dict], text_hits: list[dict]) -> list[dict]:
    """Reciprocal-Rank-Fusion of the two branches into locator buckets.

    Raw scores are incomparable (CLIP ~0.3 vs bge ~0.7), so we rank each branch
    on its own and score by rank: rrf = 1/(RRF_K + rank). Video hits within
    FUSION_WINDOW_S seconds of each other (same video) merge into one 'moment';
    document hits merge only on the exact same page/slide. Windows where BOTH
    modalities agree get boosted — two independent signals pointing at the same
    instant is the strongest evidence (video-only by construction: documents
    never appear in the frame branch).
    """
    def ranked(hits, modality):
        out = []
        for rank, h in enumerate(hits):
            t = float(h.get("t_start") or (h.get("ms", 0) or 0) / 1000.0)
            out.append({**h, "modality": modality, "rrf": 1.0 / (RRF_K + rank),
                        "t": t, "loc": _locator_key(h)})
        return out

    windows: list[dict] = []
    # Hits arrive best-first (rrf desc), so the first hit landing in a window for
    # a given modality is that modality's best hit there.
    for h in sorted(ranked(visual_hits, "frame") + ranked(text_hits, "text"),
                    key=lambda x: x["rrf"], reverse=True):
        if h["loc"] is not None:
            w = next((w for w in windows if w["video_id"] == h["video_id"]
                      and w["loc"] == h["loc"]), None)
        else:
            w = next((w for w in windows if w["video_id"] == h["video_id"]
                      and w["loc"] is None
                      and abs(w["t"] - h["t"]) <= FUSION_WINDOW_S), None)
        if w is None:
            w = {"video_id": h["video_id"], "t": h["t"], "loc": h["loc"],
                 "rrf": 0.0, "modalities": set(), "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        slot = "frame" if h["modality"] == "frame" else "text"
        # Keep only the BEST hit per modality. Summing every hit would let a
        # burst of near-identical frames clustered in one 15s window inflate its
        # score past a genuine frame+transcript match — the bug that ranked a
        # silent frame-burst above the moment that actually answered.
        if w[slot] is None:
            w[slot] = h
    for w in windows:
        # Score = best frame + best transcript hit; ×boost when BOTH modalities
        # agree at this instant (two independent signals = strongest evidence).
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0.0) + \
                   (w["text"]["rrf"] if w["text"] else 0.0)
        if {"frame", "text"} <= w["modalities"]:
            w["rrf"] *= CROSS_MODAL_BOOST
    windows.sort(key=lambda w: w["rrf"], reverse=True)
    return windows


def _deeplink(video: dict | None, video_id: str, ms: int) -> str:
    secs = ms // 1000
    if video and video.get("source") == "youtube" and video.get("url"):
        sep = "&" if "?" in video["url"] else "?"
        return f"{video['url']}{sep}t={secs}"
    return f"/api/video/{video_id}#t={secs}"


def _thumb_url(user_id: str, video_id: str, idx: int) -> str:
    """Browser-facing thumbnail URL. Presigned GET straight to the bucket when
    the provider supports it (an <img> tag can't send auth headers); the API
    serves the bytes itself only in local-dev mode."""
    if storage.presign_capable():
        return storage.presign_get(storage.frame_key(user_id, video_id, idx))
    return f"/api/frame/{video_id}/{idx:06d}.jpg?u={user_id}"


def _media_url(video: dict | None, user_id: str, video_id: str) -> str | None:
    """Playback URL for uploaded videos (YouTube plays via its own URL)."""
    if not video or video.get("source") != "upload" or not video.get("storage_key"):
        return None
    if storage.presign_capable():
        return storage.presign_get(video["storage_key"])
    return f"/api/video/{video_id}?u={user_id}"


def _page_thumb_url(user_id: str, doc_id: str, page: int) -> str | None:
    """Page/slide render URL. PDF pages always have renders; PPTX slides only
    when they embed a picture — so check existence rather than serve a 404."""
    key = storage.page_key(user_id, doc_id, page)
    try:
        if not storage.exists(key):
            return None
    except Exception:
        return None
    if storage.presign_capable():
        return storage.presign_get(key)
    return f"/api/page/{doc_id}/{page:06d}.jpg?u={user_id}"


def _doc_deeplink(meta: dict | None, doc_id: str, loc_type: str, loc_n: int) -> str:
    """Deep link into the document. Browsers' PDF viewers honor #page=N; for a
    source URL (e.g. arxiv) link there, else serve our stored copy."""
    if meta and meta.get("url"):
        return f"{meta['url']}#page={loc_n}"
    return f"/api/doc/{doc_id}#page={loc_n}"


def retrieve(question: str, user_id: str, *, top_k: int | None = None,
             video_id: str | None = None,
             video_ids: list[str] | None = None) -> dict[str, Any]:
    """Multimodal retrieve: query BOTH branches (CLIP frames + transcript text),
    fuse by RRF into time windows, and return numbered moment-citations.

    Returns {citations, best_visual, best_text} — the two raw bests feed the
    confidence gate (RRF scores are too small to threshold on). video_ids scopes
    the search to chosen videos (UI select/unselect)."""
    k = top_k or TOP_K

    # Visual branch — CLIP text→image.
    vhits = vector_store.search(embed_text(question), user_id, top_k=BRANCH_TOP_K,
                                video_id=video_id, video_ids=video_ids)
    best_visual = vhits[0]["score"] if vhits else 0.0

    # Text branch — bge query→chunk (video transcripts + paper/deck chunks).
    # Kind intent: "the slide about X" / "what does the paper say" names the
    # source type the user wants, so scope the text branch to it — otherwise
    # transcript chunks (conversational text, closest to question phrasing)
    # crowd every document out of the top ranks. Falls back to unfiltered when
    # the corpus has nothing of that kind, so intent never empties results.
    thits: list[dict] = []
    best_text = 0.0
    if config.ENABLE_TRANSCRIPT:
        qvec = embed_query(question)
        kinds = _kind_intent(question)
        if kinds:
            thits = vector_store.search_text(qvec, user_id, top_k=BRANCH_TOP_K,
                                             video_id=video_id,
                                             video_ids=video_ids, kinds=kinds)
        if not thits:
            thits = vector_store.search_text(qvec, user_id, top_k=BRANCH_TOP_K,
                                             video_id=video_id,
                                             video_ids=video_ids)
        best_text = thits[0]["score"] if thits else 0.0

    windows_all = _fuse(vhits, thits)
    windows = windows_all[:k]
    # Cross-source coverage reserve. Branch scores are fused by rank, but video
    # transcript chunks routinely out-rank document chunks for spoken-language
    # questions, which can push every paper/deck hit below the top-k cut even
    # when retrieval found them. Product stance (documented in the writeup):
    # when a source KIND was retrieved at all, its best moment deserves one of
    # the tail slots — a cross-source answer should show the paper page and the
    # slide, not six near-identical video moments. Only tail slots are
    # sacrificed and only for kinds with a real retrieved hit, so nothing is
    # invented and the head of the ranking is untouched.
    def _wkind(w) -> str:
        return ((w["text"] or w["frame"] or {}).get("kind")) or "video"

    if len(windows) == k:
        present = {_wkind(w) for w in windows}
        spare = k - 1
        for w in windows_all[k:]:
            kw = _wkind(w)
            if kw in present or spare < k // 2:
                continue
            windows[spare] = w
            present.add(kw)
            spare -= 1
    videos = db.videos_by_ids(sorted({w["video_id"] for w in windows}))
    citations = []
    for i, w in enumerate(windows, 1):
        vid = w["video_id"]
        meta = videos.get(vid)
        fr, tx = w["frame"], w["text"]
        title = (meta or {}).get("title") or vid
        # Kind: Postgres row first (legacy Qdrant points predate the payload
        # field), then the payload, defaulting to video.
        kind = ((meta or {}).get("kind")
                or (tx or fr or {}).get("kind") or "video")
        cite = {
            "n": i,
            "video_id": vid,
            "sourceId": vid,
            "kind": kind,
            "title": title,
            "url": (meta or {}).get("url"),
            "source": (meta or {}).get("source"),
            "media_url": _media_url(meta, user_id, vid),
            "score": round(w["rrf"], 4),
            "transcript": (tx or {}).get("text"),
            "modalities": sorted(w["modalities"]),
        }
        if w["loc"] is not None:
            # Document moment: locator is the page/slide stored with the chunk
            # at ingest — never derived at query time, never invented.
            loc_type, loc_n = w["loc"]
            cite.update({
                "ms": None, "timestamp": None, "idx": None,
                "locator": {loc_type: loc_n},
                "locator_label": f"p. {loc_n}" if loc_type == "page" else f"slide {loc_n}",
                "text": (tx or {}).get("text") or f"{loc_type} {loc_n} of {title}",
                "thumbnail": _page_thumb_url(user_id, vid, loc_n),
                "deeplink": _doc_deeplink(meta, vid, loc_type, loc_n),
            })
        else:
            # Video moment: anchor on the frame's exact timestamp when there is
            # one (precise visual seek); otherwise the transcript chunk's start.
            ms = int(fr["ms"]) if fr else int(w["t"] * 1000)
            t_end = int(float((fr or tx or {}).get("t_end") or ms / 1000.0) * 1000)
            idx = int(fr["idx"]) if fr else None
            cite.update({
                "ms": ms,
                "timestamp": _seconds(ms),
                "idx": idx,
                "locator": {"start_ms": ms, "end_ms": max(t_end, ms)},
                "locator_label": _seconds(ms),
                "text": (tx or {}).get("text") or f"Frame at {_seconds(ms)} in {title}",
                "thumbnail": _thumb_url(user_id, vid, idx) if idx is not None else None,
                "deeplink": _deeplink(meta, vid, ms),
            })
        citations.append(cite)
    return {"citations": citations, "best_visual": best_visual, "best_text": best_text}


def extractive_answer(citations: list[dict[str, Any]]) -> str:
    """Grounded no-LLM answer: the best retrieved excerpts, cited. This is the
    /ask_stream default — every quote and locator comes straight from
    retrieval, so nothing here can be invented."""
    parts = []
    for c in citations[:3]:
        quote = (c.get("text") or "").replace("\n", " ").strip()
        if len(quote) > 220:
            quote = quote[:220].rsplit(" ", 1)[0] + "…"
        parts.append(f'{c["title"]} ({c["locator_label"]}): "{quote}" [{c["n"]}]')
    return "Top matches — " + " · ".join(parts)


def stream_events(question: str, user_id: str, *, top_k: int | None = None,
                  video_ids: list[str] | None = None, use_llm: bool = False):
    """The /ask_stream event sequence: trace -> citations -> answer -> done.

    Default (use_llm=False) answers extractively from retrieval — fast,
    deterministic, entirely our own latency (this is what the SLA gate times).
    use_llm=True (the UI and demo) adds full LLM synthesis over the same
    citations. Citations are identical in both modes.
    """
    import time as _time

    t0 = _time.perf_counter()
    r = retrieve(question, user_id, top_k=top_k, video_ids=video_ids)
    citations = r["citations"]
    yield {"trace": {"stage": "retrieval",
                     "ms": round((_time.perf_counter() - t0) * 1000, 1),
                     "results": len(citations),
                     "kinds": sorted({c["kind"] for c in citations}),
                     "best_visual": round(r["best_visual"], 3),
                     "best_text": round(r["best_text"], 3)}}
    yield {"citations": citations}

    if not citations:
        yield {"answer": "No relevant moments were found. Try ingesting a source first.",
               "llm_used": False, "abstained": True}
    else:
        visual_ok = r["best_visual"] >= CONFIDENCE_THRESHOLD
        text_ok = r["best_text"] >= TEXT_CONFIDENCE_THRESHOLD
        if CONFIDENCE_THRESHOLD and not visual_ok and not text_ok:
            yield {"answer": ABSTAIN, "llm_used": False, "abstained": True}
        else:
            cfg, source = resolve_llm(user_id) if use_llm else (None, "skipped")
            if cfg is not None:
                moments = _build_moments(user_id, citations)
                ans = _validate_citations(llm.answer(question, moments, cfg),
                                          len(citations))
                yield {"answer": ans, "llm_used": True, "llm_source": source,
                       "llm_model": cfg.model}
            else:
                yield {"answer": extractive_answer(citations), "llm_used": False}
    yield {"done": True}


def _fallback_answer(citations: list[dict[str, Any]]) -> str:
    """No-LLM summary: rank the visually-closest moments. Honest about being
    similarity, not synthesis."""
    top = citations[0]
    where = f"{top['title']} at {top['timestamp']}" if top.get("title") else top["timestamp"]
    others = ", ".join(f"{c['timestamp']} [{c['n']}]" for c in citations[1:4])
    msg = f"Closest visual match: {where} [{top['n']}] (similarity {top['score']})."
    if others:
        msg += f" Other relevant moments: {others}."
    return msg


_CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def _validate_citations(answer: str, n_frames: int) -> str:
    """Strip invented [n] references the model has no frame for."""
    def fix(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"\s*,\s*", m.group(1))]
        valid = [str(x) for x in nums if 1 <= x <= n_frames]
        return f"[{', '.join(valid)}]" if valid else ""
    return _CITE_RE.sub(fix, answer)


def _build_moments(user_id: str, citations: list[dict[str, Any]]) -> list[dict]:
    """Turn citations into what the LLM sees: each moment carries its image
    (video frame or page/slide render, if any), its text excerpt, and a
    kind-appropriate label ("@ 14:13" | "p. 4 of Title" | "slide 12 of Title")
    so the model can talk about pages and slides, not just timestamps."""
    def image_bytes(c):
        try:
            if c.get("idx") is not None:
                return storage.get_bytes(storage.frame_key(user_id, c["video_id"], c["idx"]))
            loc = c.get("locator") or {}
            page = loc.get("page") or loc.get("slide")
            if page and c.get("thumbnail"):
                return storage.get_bytes(storage.page_key(user_id, c["video_id"], page))
        except Exception:
            return None
        return None

    def label(c):
        if c.get("kind") in ("paper", "deck"):
            return f"{c.get('locator_label')} of {c.get('title')}"
        return f"@ {c.get('timestamp')}"

    with ThreadPoolExecutor(max_workers=6) as ex:
        images = list(ex.map(image_bytes, citations))
    return [{"image": img, "transcript": c.get("transcript") or c.get("text"),
             "timestamp": c.get("timestamp") or "", "label": label(c)}
            for img, c in zip(images, citations)]


def resolve_llm(user_id: str) -> tuple[llm.LLMConfig | None, str]:
    """Which model answers for this tenant: their own hosted endpoint
    (ms_user_llms — e.g. a vLLM server) first, the server-wide LLM_* env
    config as fallback. Returns (config, source) with source in
    {"user", "server", "none"}."""
    row = db.get_user_llm(user_id)
    if row and row.get("model"):
        return llm.from_row(row), "user"
    cfg = llm.env_config()
    return (cfg, "server") if cfg else (None, "none")


def ask(question: str, user_id: str, *, top_k: int | None = None,
        video_id: str | None = None,
        video_ids: list[str] | None = None) -> dict[str, Any]:
    r = retrieve(question, user_id, top_k=top_k, video_id=video_id, video_ids=video_ids)
    citations = r["citations"]
    result: dict[str, Any] = {"question": question, "citations": citations}

    if not citations:
        result.update(answer="No relevant moments were found. Try ingesting a video first.",
                      llm_used=False, abstained=True)
        return result

    # Gate 1 — confidence on the RAW per-branch bests (not the RRF score).
    # Abstain only if NEITHER what's on screen nor what's said looks relevant.
    visual_ok = r["best_visual"] >= CONFIDENCE_THRESHOLD
    text_ok = r["best_text"] >= TEXT_CONFIDENCE_THRESHOLD
    if CONFIDENCE_THRESHOLD and not visual_ok and not text_ok:
        result.update(answer=ABSTAIN, llm_used=False, abstained=True)
        return result

    cfg, source = resolve_llm(user_id)
    if cfg is None:
        # No generative model — summarize the best matches instead of inventing.
        result.update(answer=_fallback_answer(citations), llm_used=False,
                      note=("Retrieval-only results. Connect your own model "
                            "(vLLM/Ollama/API) in settings, or set LLM_API_KEY "
                            "on the server, for a synthesized, grounded answer."))
        return result

    moments = _build_moments(user_id, citations)
    result["answer"] = _validate_citations(llm.answer(question, moments, cfg),
                                           len(citations))
    result["llm_used"] = True
    result["llm_source"] = source          # "user" = their own hosted model
    result["llm_model"] = cfg.model
    return result
