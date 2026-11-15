
import json
from pathlib import Path

from src.preprocessing.chunker import chunk_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_chunks.json"
)


def run_chunking():

    print("\nStarting document chunking...")

    # Load extracted document
    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        document = json.load(file)

    # Generate chunks
    chunks = chunk_document(
        pages=document["pages"],
        source_file=document["document"]["source_file"],
        chunk_size=1200,
        overlap=200
    )

    # Calculate statistics
    total_chunks = len(chunks)

    chunk_sizes = [
        chunk["character_count"] for chunk in chunks
    ]

    average_size = (
        sum(chunk_sizes) / total_chunks
        if total_chunks else 0
    )

    min_size = min(chunk_sizes, default=0)
    max_size = max(chunk_sizes, default=0)

    # Create output
    output = {
        "document": document["document"],
        "chunking_configuration": {
            "chunk_size": 1200,
            "overlap": 200,
            "strategy": "page_based_character_chunking"
        },
        "statistics": {
            "total_chunks": total_chunks,
            "average_chunk_size": round(average_size, 2),
            "minimum_chunk_size": min_size,
            "maximum_chunk_size": max_size
        },
        "chunks": chunks
    }

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nChunking completed successfully!")

    print(f"Total chunks: {total_chunks}")
    print(f"Average chunk size: {average_size:.2f}")
    print(f"Minimum chunk size: {min_size}")
    print(f"Maximum chunk size: {max_size}")

    print(f"\nSaved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_chunking()
