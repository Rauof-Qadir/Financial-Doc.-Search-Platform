
import pytest

from src.generation.context_builder import build_context


@pytest.fixture
def sample_results():
    return [
        {
            "chunk_id": "apple-p050-c005",
            "page_number": 50,
            "score": 0.91,
            "text": "R&D expenses in 2023 were $29,915 million.",
        },
        {
            "chunk_id": "apple-p050-c006",
            "page_number": 50,
            "score": 0.85,
            "text": "R&D expenses in 2022 were $26,251 million.",
        },
    ]


def test_build_context_includes_evidence(sample_results):
    context = build_context(sample_results)

    assert "SOURCE 1" in context
    assert "apple-p050-c005" in context
    assert "Page: 50" in context
    assert "$29,915 million" in context


def test_build_context_respects_max_chunks(sample_results):
    context = build_context(sample_results, max_chunks=1)

    assert "SOURCE 1" in context
    assert "SOURCE 2" not in context


def test_build_context_handles_empty_results():
    context = build_context([])

    assert context == "No supporting evidence was retrieved."


def test_build_context_rejects_invalid_max_chunks(sample_results):
    with pytest.raises(ValueError):
        build_context(sample_results, max_chunks=0)


def test_build_context_rejects_missing_fields():
    results = [{"chunk_id": "chunk-001"}]

    with pytest.raises(ValueError):
        build_context(results)


def test_build_context_rejects_empty_text():
    results = [
        {
            "chunk_id": "chunk-001",
            "page_number": 10,
            "text": "   ",
        }
    ]

    with pytest.raises(ValueError):
        build_context(results)
