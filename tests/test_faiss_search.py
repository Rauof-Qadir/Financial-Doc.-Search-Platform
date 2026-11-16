import numpy as np
import pytest

from src.retrieval.faiss_search import FAISSSearchEngine


class FakeIndex:
    ntotal = 2

    def search(self, query, k):
        return (
            [[0.91, 0.82][:k]],
            [[0, 1][:k]],
        )


class FakeModel:
    def encode(self, queries, **kwargs):
        return np.array([[0.1] * 384], dtype=np.float32)


def make_engine():
    engine = FAISSSearchEngine.__new__(FAISSSearchEngine)

    engine.index = FakeIndex()
    engine.metadata = [
        {
            "chunk_id": "chunk-001",
            "page_number": 10,
            "text": "Research and development expenses.",
        },
        {
            "chunk_id": "chunk-002",
            "page_number": 20,
            "text": "Apple revenue information.",
        },
    ]
    engine.model = FakeModel()

    return engine


def test_search_returns_results():
    engine = make_engine()

    results = engine.search("R&D expenses", top_k=2)

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-001"
    assert results[0]["score"] == pytest.approx(0.91)
    assert results[1]["page_number"] == 20


def test_search_rejects_empty_query():
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.search("   ")


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_search_rejects_invalid_top_k(top_k):
    engine = make_engine()

    with pytest.raises(ValueError):
        engine.search("Apple revenue", top_k=top_k)


def test_search_limits_results_to_available_chunks():
    engine = make_engine()

    results = engine.search("Apple", top_k=10)

    assert len(results) == 2
