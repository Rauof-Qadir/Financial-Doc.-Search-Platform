
import pytest

from src.generation.rag_pipeline import FinancialRAGPipeline


@pytest.fixture
def sample_metadata():
    return [
        {
            "chunk_id": "apple-p050-c005",
            "page_number": 50,
            "text": (
                "Research and development expenses were "
                "$29,915 million in 2023."
            ),
        },
        {
            "chunk_id": "apple-p050-c006",
            "page_number": 50,
            "text": (
                "Research and development expenses were "
                "$26,251 million in 2022."
            ),
        },
        {
            "chunk_id": "apple-p020-c001",
            "page_number": 20,
            "text": "Apple sells products around the world.",
        },
    ]


def test_pipeline_retrieves_relevant_evidence(sample_metadata):
    pipeline = FinancialRAGPipeline(sample_metadata)

    result = pipeline.run("What were R&D expenses in 2023?")

    assert result["evidence_found"] is True
    assert len(result["retrieved_chunks"]) > 0
    assert "$29,915 million" in result["context"]


def test_pipeline_preserves_source_information(sample_metadata):
    pipeline = FinancialRAGPipeline(sample_metadata)

    result = pipeline.run("R&D expenses in 2023")

    assert "apple-p050-c005" in result["context"]
    assert "Page: 50" in result["context"]


def test_pipeline_returns_no_evidence_for_unknown_query(sample_metadata):
    pipeline = FinancialRAGPipeline(sample_metadata)

    result = pipeline.run("volcanic activity on Mars")

    assert result["evidence_found"] is False
    assert result["retrieved_chunks"] == []
    assert result["context"] == "No supporting evidence was retrieved."


def test_pipeline_rejects_empty_query(sample_metadata):
    pipeline = FinancialRAGPipeline(sample_metadata)

    with pytest.raises(ValueError):
        pipeline.run("   ")


def test_pipeline_rejects_invalid_top_k(sample_metadata):
    with pytest.raises(ValueError):
        FinancialRAGPipeline(sample_metadata, top_k=0)


def test_pipeline_rejects_empty_metadata():
    with pytest.raises(ValueError):
        FinancialRAGPipeline([])
