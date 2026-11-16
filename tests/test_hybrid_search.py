
import pytest

from src.retrieval import hybrid_search


def test_tokenize_converts_text_to_lowercase_tokens():
    result = hybrid_search.tokenize("Apple Revenue 2023")

    assert result == ["apple", "revenue", "2023"]


def test_tokenize_rejects_non_string():
    with pytest.raises(TypeError):
        hybrid_search.tokenize(None)


def test_rrf_combines_rankings_and_prioritizes_shared_results():
    semantic = [
        {"chunk_id": "A", "text": "Document A"},
        {"chunk_id": "B", "text": "Document B"},
        {"chunk_id": "C", "text": "Document C"},
    ]

    keyword = [
        {"chunk_id": "B", "text": "Document B", "bm25_score": 2.0},
        {"chunk_id": "C", "text": "Document C", "bm25_score": 1.0},
        {"chunk_id": "D", "text": "Document D", "bm25_score": 0.5},
    ]

    results = hybrid_search.reciprocal_rank_fusion(
        [semantic, keyword]
    )

    assert results[0]["chunk_id"] == "B"
    assert results[0]["rrf_score"] > results[1]["rrf_score"]


def test_rrf_does_not_duplicate_chunks():
    semantic = [
        {"chunk_id": "A"},
        {"chunk_id": "B"},
    ]

    keyword = [
        {"chunk_id": "A"},
        {"chunk_id": "C"},
    ]

    results = hybrid_search.reciprocal_rank_fusion(
        [semantic, keyword]
    )

    result_ids = [item["chunk_id"] for item in results]

    assert len(result_ids) == 3
    assert len(result_ids) == len(set(result_ids))


def test_rrf_rejects_invalid_rrf_k():
    with pytest.raises(ValueError):
        hybrid_search.reciprocal_rank_fusion(
            [],
            rrf_k=0,
        )


def test_rrf_rejects_missing_chunk_id():
    with pytest.raises(ValueError):
        hybrid_search.reciprocal_rank_fusion(
            [[{"text": "Missing ID"}]]
        )


def test_hybrid_search_returns_ranked_results(monkeypatch):
    fake_metadata = [
        {
            "chunk_id": "chunk-1",
            "text": "Apple revenue increased in 2023.",
            "source_file": "apple.pdf",
            "page_number": 1,
        },
        {
            "chunk_id": "chunk-2",
            "text": "Research and development expenses.",
            "source_file": "apple.pdf",
            "page_number": 2,
        },
        {
            "chunk_id": "chunk-3",
            "text": "Company operations and business risks.",
            "source_file": "apple.pdf",
            "page_number": 3,
        },
    ]

    class FakeSemanticSearchEngine:
        def __init__(self, vector_store_file=None):
            self.metadata = fake_metadata

        def search(self, query, top_k=5):
            return [
                {
                    **fake_metadata[0],
                    "similarity_score": 0.9,
                },
                {
                    **fake_metadata[1],
                    "similarity_score": 0.7,
                },
                {
                    **fake_metadata[2],
                    "similarity_score": 0.5,
                },
            ][:top_k]

    monkeypatch.setattr(
        hybrid_search,
        "SemanticSearchEngine",
        FakeSemanticSearchEngine,
    )

    engine = hybrid_search.HybridSearchEngine()

    results = engine.search(
        "Apple revenue",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert "rrf_score" in results[0]
    assert "similarity_score" in results[0]
    assert "bm25_score" in results[0]
    assert "bm25_rank" in results[0]



def test_hybrid_search_rejects_empty_query(monkeypatch):
    class FakeSemanticSearchEngine:
        def __init__(self, vector_store_file=None):
            self.metadata = [
                {
                    "chunk_id": "chunk-1",
                    "text": "Apple revenue increased.",
                }
            ]

        def search(self, query, top_k=5):
            return []

    monkeypatch.setattr(
        hybrid_search,
        "SemanticSearchEngine",
        FakeSemanticSearchEngine,
    )

    engine = hybrid_search.HybridSearchEngine()

    with pytest.raises(ValueError):
        engine.search("   ")


def test_hybrid_search_rejects_query_without_searchable_tokens(monkeypatch):
    class FakeSemanticSearchEngine:
        def __init__(self, vector_store_file=None):
            self.metadata = [
                {
                    "chunk_id": "chunk-1",
                    "text": "Apple revenue increased.",
                }
            ]

        def search(self, query, top_k=5):
            return []

    monkeypatch.setattr(
        hybrid_search,
        "SemanticSearchEngine",
        FakeSemanticSearchEngine,
    )

    engine = hybrid_search.HybridSearchEngine()

    with pytest.raises(ValueError):
        engine.search("!!!")
