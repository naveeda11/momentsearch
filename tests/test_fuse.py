"""Fusion regressions: documents bucket by locator, video by time window."""
from src.rag.search import _fuse, _kind_intent, _locator_key


def _hit(video_id, *, page=None, slide=None, t_start=0.0, text="x", kind=None):
    h = {"video_id": video_id, "t_start": t_start, "text": text}
    if page is not None:
        h["page"] = page
    if slide is not None:
        h["slide"] = slide
    if kind:
        h["kind"] = kind
    return h


def test_doc_chunks_at_t0_do_not_collapse():
    # Every doc chunk carries t=0; time-windowing would have merged the whole
    # paper into one citation. Locator bucketing must keep pages distinct.
    hits = [_hit("doc_a", page=1), _hit("doc_a", page=2), _hit("doc_a", page=7)]
    windows = _fuse([], hits)
    assert len(windows) == 3
    assert {w["loc"] for w in windows} == {("page", 1), ("page", 2), ("page", 7)}


def test_same_page_chunks_merge():
    hits = [_hit("doc_a", page=4, text="c1"), _hit("doc_a", page=4, text="c2")]
    windows = _fuse([], hits)
    assert len(windows) == 1
    assert windows[0]["loc"] == ("page", 4)


def test_slide_and_page_never_merge():
    hits = [_hit("doc_a", page=3), _hit("doc_b", slide=3)]
    assert len(_fuse([], hits)) == 2


def test_video_time_window_preserved():
    frames = [_hit("yt_v", t_start=100.0), _hit("yt_v", t_start=110.0)]
    text = [_hit("yt_v", t_start=105.0)]
    windows = _fuse(frames, text)
    assert len(windows) == 1  # all within 15s of each other
    assert windows[0]["loc"] is None


def test_cross_modal_boost_video_only():
    # A frame+text agreement at the same instant outscores two text-only pages
    # with the same ranks.
    frames = [_hit("yt_v", t_start=50.0)]
    text = [_hit("yt_v", t_start=52.0), _hit("doc_a", page=1)]
    windows = _fuse(frames, text)
    video = next(w for w in windows if w["video_id"] == "yt_v")
    doc = next(w for w in windows if w["video_id"] == "doc_a")
    assert {"frame", "text"} <= video["modalities"]
    assert video["rrf"] > doc["rrf"]


def test_locator_key():
    assert _locator_key({"page": 4}) == ("page", 4)
    assert _locator_key({"slide": 12}) == ("slide", 12)
    assert _locator_key({"t_start": 3.0}) is None


def test_kind_intent():
    assert _kind_intent("the slide about one index for every source") == ["deck"]
    assert _kind_intent("what does the survey say about hybrid retrieval") == ["paper"]
    assert _kind_intent("show me the deck and the paper") == ["deck", "paper"]
    assert _kind_intent("how does attention work") is None
