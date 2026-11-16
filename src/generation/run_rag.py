
import argparse
import json
from pathlib import Path

from src.generation.rag_pipeline import FinancialRAGPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNK_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_chunks_token_aware.json"
)


def load_document_chunks() -> list[dict]:
    """Load preprocessed financial document chunks."""

    if not CHUNK_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNK_FILE}"
        )

    data = json.loads(CHUNK_FILE.read_text(encoding="utf-8"))

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("Document contains no valid chunks.")

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Run evidence retrieval on Apple's annual report."
    )

    parser.add_argument(
        "--query",
        type=str,
        default="What were Apple's research and development expenses in 2023?",
        help="Financial question to search in the annual report.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of evidence chunks to retrieve.",
    )

    args = parser.parse_args()

    # Load all 347 preprocessed document chunks.
    chunks = load_document_chunks()

    # Initialize the retrieval and context-building pipeline.
    pipeline = FinancialRAGPipeline(
        metadata=chunks,
        top_k=args.top_k,
    )

    # Retrieve evidence for the user's financial question.
    result = pipeline.run(args.query)

    print("\n" + "=" * 65)
    print("FINANCIAL DOCUMENT RAG — EVIDENCE RETRIEVAL")
    print("=" * 65)

    print(f"\nQuestion: {result['query']}")
    print(f"Document: Apple 2023 Annual Report")
    print(f"Total indexed chunks: {len(chunks)}")
    print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")
    print(f"Evidence found: {result['evidence_found']}")

    print("\n" + "-" * 65)
    print("RETRIEVED EVIDENCE")
    print("-" * 65)

    print(result["context"])

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
