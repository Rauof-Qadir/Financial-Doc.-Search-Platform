
import pytest

from src.retrieval.bm25_search import BM25SearchEngine


@pytest.fixture
def metadata():
    return [
        {
            "chunk_id": "chunk-1",
            "text": "Apple revenue increased during the year.",
            "page_number": 1,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Research and development expenses increased.",
            "page_number": 2,
        },
        {
            "chunk_id": "chunk-3",
            "text": "Apple reported revenue and net sales.",
            "page_number": 3,
        },
    ]


def test_bm25_returns_relevant_results(metadata):
    engine = BM25SearchEngine(metadata)

    results = engine.search("Apple revenue", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] in {"chunk-1", "chunk-3"}
    assert "bm25_score" in results[0]


def test_bm25_rejects_empty_metadata():
    with pytest.raises(ValueError):
        BM25SearchEngine([])


def test_bm25_rejects_empty_query(metadata):
    engine = BM25SearchEngine(metadata)

    with pytest.raises(ValueError):
        engine.search("   ")


def test_bm25_rejects_invalid_top_k(metadata):
    engine = BM25SearchEngine(metadata)

    with pytest.raises(ValueError):
        engine.search("Apple revenue", top_k=0)


def test_bm25_returns_no_results_for_unknown_terms(metadata):
    engine = BM25SearchEngine(metadata)

    results = engine.search("quantum spacecraft", top_k=5)

    assert results == []
