
import json
from pathlib import Path

from src.ingestion.pdf_loader import extract_pdf


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Input and output paths
INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "apple_annual_report.pdf"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report.json"
)


def run_ingestion():

    print("\nStarting document ingestion...")

    # Extract PDF content
    result = extract_pdf(INPUT_FILE)

    pages = result["pages"]

    # Calculate extraction statistics
    non_empty_pages = sum(
        1 for page in pages if page["text"].strip()
    )

    empty_pages = sum(
        1 for page in pages if not page["text"].strip()
    )

    total_characters = len(result["full_text"])

    total_words = len(result["full_text"].split())

    # Create structured output
    report = {
        "document": {
            "source_file": result["source_file"],
            "total_pages": result["total_pages"],
            "metadata": result["metadata"]
        },
        "extraction_statistics": {
            "non_empty_pages": non_empty_pages,
            "empty_pages": empty_pages,
            "total_characters": total_characters,
            "total_words": total_words
        },
        "pages": pages,
        "full_text": result["full_text"]
    }

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save JSON report
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nIngestion completed successfully!")

    print(f"Document: {result['source_file']}")
    print(f"Total pages: {result['total_pages']}")
    print(f"Non-empty pages: {non_empty_pages}")
    print(f"Empty pages: {empty_pages}")
    print(f"Total words: {total_words}")
    print(f"Total characters: {total_characters}")

    print(f"\nOutput saved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_ingestion()
