"""Chunking: page boundaries are never crossed; decks are one chunk per slide."""
from src import config
from src.ingest.documents import PageRec, chunk_pages, full_text


def test_paper_chunks_never_cross_pages():
    long_text = "word " * 800  # ~4000 chars, several chunks
    pages = [PageRec(page=1, text=long_text), PageRec(page=2, text=long_text)]
    chunks = chunk_pages(pages, "paper")
    assert len(chunks) > 2
    for c in chunks:
        assert c["page"] in (1, 2)
        assert len(c["text"]) <= config.DOC_CHUNK_CHARS


def test_paper_chunk_overlap():
    text = "abcdefghij" * 200  # 2000 chars -> 2+ chunks with overlap
    chunks = chunk_pages([PageRec(page=1, text=text)], "paper")
    assert len(chunks) >= 2
    # The second chunk starts DOC_CHUNK_OVERLAP chars before the first ended.
    step = config.DOC_CHUNK_CHARS - config.DOC_CHUNK_OVERLAP
    assert chunks[1]["text"][:20] == text[step:step + 20]


def test_deck_one_chunk_per_slide():
    pages = [PageRec(page=1, text="Title slide"),
             PageRec(page=2, text="Agenda " * 400),  # long slide still one chunk
             PageRec(page=3, text="")]               # empty slide -> dropped
    chunks = chunk_pages(pages, "deck")
    assert [c["slide"] for c in chunks] == [1, 2]


def test_caption_merged_into_embedded_text():
    p = PageRec(page=5, text="", caption="A bar chart of recall by method.",
                needs_caption=True)
    assert "[Figure] A bar chart" in full_text(p)
    chunks = chunk_pages([p], "paper")
    assert chunks and chunks[0]["page"] == 5
