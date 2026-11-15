
import pytest

from src.preprocessing.chunker import (
    chunk_text,
    chunk_document
)


def test_empty_text_returns_no_chunks():

    assert chunk_text("") == []


def test_short_text_returns_one_chunk():

    text = "Apple financial report"

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_creates_multiple_chunks():

    text = "financial information " * 200

    chunks = chunk_text(
        text,
        chunk_size=1200,
        overlap=200
    )

    assert len(chunks) > 1


def test_invalid_chunk_size():

    with pytest.raises(ValueError):
        chunk_text("financial report", chunk_size=0)


def test_invalid_overlap():

    with pytest.raises(ValueError):
        chunk_text(
            "financial report",
            chunk_size=100,
            overlap=100
        )


def test_document_metadata_is_preserved():

    pages = [
        {
            "page_number": 5,
            "text": "Apple financial report " * 100
        }
    ]

    chunks = chunk_document(
        pages,
        source_file="apple_annual_report.pdf"
    )

    assert len(chunks) > 0

    assert chunks[0]["source_file"] == (
        "apple_annual_report.pdf"
    )

    assert chunks[0]["page_number"] == 5

    assert "chunk_id" in chunks[0]
