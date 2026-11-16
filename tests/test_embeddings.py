
import json

import numpy as np
import pytest

from src.embeddings.embedding_generator import (
    load_chunks,
    save_embeddings,
)


def sample_chunks():
    return [
        {
            "chunk_id": "apple-p001-c001",
            "source_file": "apple_annual_report.pdf",
            "page_number": 1,
            "chunk_index": 1,
            "text": "Apple reported strong revenue growth."
        },
        {
            "chunk_id": "apple-p002-c001",
            "source_file": "apple_annual_report.pdf",
            "page_number": 2,
            "chunk_index": 1,
            "text": "Research and development expenses increased."
        }
    ]


def test_load_chunks_successfully(tmp_path):
    file_path = tmp_path / "chunks.json"

    file_path.write_text(
        json.dumps(sample_chunks()),
        encoding="utf-8"
    )

    chunks = load_chunks(file_path)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "apple-p001-c001"


def test_missing_chunk_file():
    with pytest.raises(FileNotFoundError):
        load_chunks("missing_chunks.json")


def test_empty_chunk_file(tmp_path):
    file_path = tmp_path / "chunks.json"
    file_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError):
        load_chunks(file_path)


def test_empty_chunk_text(tmp_path):
    chunks = sample_chunks()
    chunks[0]["text"] = "   "

    file_path = tmp_path / "chunks.json"
    file_path.write_text(
        json.dumps(chunks),
        encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_chunks(file_path)


def test_save_embeddings_successfully(tmp_path):
    chunks = sample_chunks()

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=np.float32
    )

    output_file = tmp_path / "embeddings.npz"

    save_embeddings(chunks, embeddings, output_file)

    assert output_file.exists()

    with np.load(output_file) as data:
        saved_embeddings = data["embeddings"]
        metadata = json.loads(str(data["metadata"]))

    assert saved_embeddings.shape == (2, 3)
    assert len(metadata) == 2
    assert metadata[0]["chunk_id"] == "apple-p001-c001"
    assert metadata[1]["page_number"] == 2


def test_mismatched_chunk_and_embedding_count(tmp_path):
    chunks = sample_chunks()
    embeddings = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(ValueError):
        save_embeddings(
            chunks,
            embeddings,
            tmp_path / "embeddings.npz"
        )
