import json
from pathlib import Path

from src.generation.financial_qa import FinancialQuestionAnsweringPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_chunks_token_aware.json"
)


def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Document chunks not found: {CHUNKS_PATH}"
        )

    data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("No document chunks found.")

    return chunks


def main():
    chunks = load_chunks()

    pipeline = FinancialQuestionAnsweringPipeline(
        metadata=chunks,
        top_k=5,
    )

    question = input("\nAsk a question about Apple's 2023 annual report: ")

    result = pipeline.answer(question)

    print("\n" + "=" * 65)
    print("FINANCIAL DOCUMENT INTELLIGENCE")
    print("=" * 65)

    print("\nQUESTION:")
    print(result["query"])

    print("\nANSWER:")
    print(result["answer"])

    print("\nEVIDENCE FOUND:", result["evidence_found"])
    print("MODEL:", result["model"])

    print("\nCITATION VALIDATION:")
    print(json.dumps(
         result["citation_validation"],
         indent=4
    ))

    print("\nRETRIEVED SOURCES:")

    for source in result["sources"]:
        print(
            f"- Page {source['page_number']} "
            f"| Chunk: {source['chunk_id']}"
        )

    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
