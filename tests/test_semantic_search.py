
import json

import numpy as np
import pytest

from src.retrieval.semantic_search import (
    load_vector_store,
    rank_chunks
)


def sample_metadata():
    return [
        {"chunk_id": "chunk-1", "page_number": 1, "text": "Apple research expenses"},
        {"chunk_id": "chunk-2", "page_number": 2, "text": "Apple revenue"},
        {"chunk_id": "chunk-3", "page_number": 3, "text": "Apple operating costs"}
    ]


def test_load_vector_store_successfully(tmp_path):
    file_path = tmp_path / "vectors.npz"

    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2]
        ],
        dtype=np.float32
    )

    metadata = sample_metadata()

    np.savez_compressed(
        file_path,
        embeddings=embeddings,
        metadata=np.array(json.dumps(metadata), dtype=np.str_)
    )

    loaded_embeddings, loaded_metadata = load_vector_store(file_path)

    assert loaded_embeddings.shape == (3, 2)
    assert len(loaded_metadata) == 3
    assert loaded_metadata[0]["chunk_id"] == "chunk-1"


def test_missing_vector_store():
    with pytest.raises(FileNotFoundError):
        load_vector_store("missing_vectors.npz")


def test_rank_chunks_returns_top_results():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2]
        ],
        dtype=np.float32
    )

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = rank_chunks(
        query,
        embeddings,
        sample_metadata(),
        top_k=2
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "chunk-1"
    assert results[1]["chunk_id"] == "chunk-3"


def test_similarity_scores_are_valid():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.2]
        ],
        dtype=np.float32
    )

    query = np.array([1.0, 0.0], dtype=np.float32)

    results = rank_chunks(
        query,
        embeddings,
        sample_metadata(),
        top_k=3
    )

    for result in results:
        assert -1.0 <= result["similarity_score"] <= 1.0


def test_invalid_top_k():
    with pytest.raises(ValueError):
        rank_chunks(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [{"chunk_id": "chunk-1"}],
            top_k=0
        )


def test_mismatched_dimensions():
    with pytest.raises(ValueError):
        rank_chunks(
            np.array([1.0, 0.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [{"chunk_id": "chunk-1"}],
            top_k=1
        )


def test_empty_query_vector():
    with pytest.raises(ValueError):
        rank_chunks(
            np.array([0.0, 0.0]),
            np.array([[1.0, 0.0]]),
            [{"chunk_id": "chunk-1"}],
            top_k=1
        )
