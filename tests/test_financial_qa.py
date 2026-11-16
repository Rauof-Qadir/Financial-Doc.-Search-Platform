from unittest.mock import Mock

import pytest

from src.generation.financial_qa import (
    FinancialQuestionAnsweringPipeline,
)


@pytest.fixture
def pipeline():
    instance = FinancialQuestionAnsweringPipeline.__new__(
        FinancialQuestionAnsweringPipeline
    )

    instance.rag = Mock()
    instance.generator = Mock()
    instance.generator.model = "test-gemini-model"

    instance.rag.run.return_value = {
        "query": "test question",
        "evidence_found": True,
        "context": (
            "SOURCE 1\n"
            "Chunk ID: apple-p023-c001\n"
            "Page: 23\n"
            "Evidence:\n"
            "Apple reported net income of $97.0 billion."
        ),
        "retrieved_chunks": [
            {
                "chunk_id": "apple-p023-c001",
                "page_number": 23,
                "text": "Apple reported net income of $97.0 billion.",
            }
        ],
    }

    return instance


def test_answer_generates_response_from_evidence(pipeline):
    pipeline.generator.generate.return_value = (
        "Apple's net income was $97.0 billion "
        "[Page: 23 | Chunk: apple-p023-c001]."
    )

    result = pipeline.answer(
        "What was Apple's net income in 2023?"
    )

    assert result["evidence_found"] is True
    assert "$97.0 billion" in result["answer"]
    assert result["model"] == "test-gemini-model"
    pipeline.generator.generate.assert_called_once()


def test_answer_preserves_retrieved_sources(pipeline):
    pipeline.generator.generate.return_value = (
        "Apple's R&D spending was $29,915 million."
    )

    result = pipeline.answer(
        "How much did Apple spend on research and development?"
    )

    assert len(result["sources"]) == 1
    assert result["sources"][0]["chunk_id"] == "apple-p023-c001"
    assert result["sources"][0]["page_number"] == 23


def test_answer_does_not_call_gemini_without_evidence(pipeline):
    pipeline.rag.run.return_value = {
        "query": "unknown question",
        "evidence_found": False,
        "context": "No supporting evidence was retrieved.",
        "retrieved_chunks": [],
    }

    result = pipeline.answer(
        "What was Apple's employee satisfaction score?"
    )

    assert result["evidence_found"] is False
    assert result["sources"] == []
    pipeline.generator.generate.assert_not_called()


def test_answer_rejects_empty_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.answer("")


def test_answer_rejects_whitespace_question(pipeline):
    with pytest.raises(ValueError):
        pipeline.answer("   ")


def test_answer_propagates_generator_error(pipeline):
    pipeline.generator.generate.side_effect = RuntimeError(
        "Gemini API unavailable"
    )

    with pytest.raises(RuntimeError):
        pipeline.answer(
            "What was Apple's net income in 2023?"
        )


def test_valid_citation_is_accepted(pipeline):
    from src.generation.financial_qa import validate_citations

    result = validate_citations(
        answer=(
            "Apple's net income was $97 billion "
            "[Page: 23 | Chunk: apple-p023-c001]."
        ),
        sources=[
            {
                "chunk_id": "apple-p023-c001",
                "page_number": 23,
            }
        ],
    )

    assert result["valid"] is True
    assert result["citations_found"] == 1
    assert result["invalid_citations"] == []


def test_unknown_chunk_citation_is_rejected(pipeline):
    from src.generation.financial_qa import validate_citations

    result = validate_citations(
        answer="Revenue was reported [Page: 23 | Chunk: fake-chunk].",
        sources=[
            {
                "chunk_id": "apple-p023-c001",
                "page_number": 23,
            }
        ],
    )

    assert result["valid"] is False
    assert len(result["invalid_citations"]) == 1


def test_wrong_page_citation_is_rejected(pipeline):
    from src.generation.financial_qa import validate_citations

    result = validate_citations(
        answer=(
            "Net income was $97 billion "
            "[Page: 99 | Chunk: apple-p023-c001]."
        ),
        sources=[
            {
                "chunk_id": "apple-p023-c001",
                "page_number": 23,
            }
        ],
    )

    assert result["valid"] is False
    assert "Page mismatch" in result["invalid_citations"][0]