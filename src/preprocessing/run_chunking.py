import json
from pathlib import Path

from transformers import AutoTokenizer

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
    / "apple_annual_report_chunks_token_aware.json"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_TOKENS = 220
OVERLAP_TOKENS = 40


def run_chunking():

    print("\nStarting token-aware document chunking...")

    print("\nLoading document...")

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        document = json.load(file)

    print("Document loaded.")

    print("\nLoading MiniLM tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # SentenceTransformer model's configured maximum sequence length.
    # Do not use tokenizer.model_max_length for this value.

    model_limit = 256

    print(f"Tokenizer loaded: {MODEL_NAME}")
    print(f"Model token limit: {model_limit}")
    print(f"Chunk token limit: {MAX_TOKENS}")

    if MAX_TOKENS > model_limit:
        raise ValueError(
            f"MAX_TOKENS ({MAX_TOKENS}) exceeds "
            f"model limit ({model_limit})."
        )

    print("\nGenerating chunks...")

    chunks = chunk_document(
        pages=document["pages"],
        source_file=document["document"]["source_file"],
        tokenizer=tokenizer,
        max_tokens=MAX_TOKENS,
        overlap=OVERLAP_TOKENS
    )

    total_chunks = len(chunks)

    token_sizes = [
        chunk["token_count"] for chunk in chunks
    ]

    character_sizes = [
        chunk["character_count"] for chunk in chunks
    ]

    average_tokens = (
        sum(token_sizes) / total_chunks
        if total_chunks else 0
    )

    average_characters = (
        sum(character_sizes) / total_chunks
        if total_chunks else 0
    )

    min_tokens = min(token_sizes, default=0)
    max_observed_tokens = max(token_sizes, default=0)

    exceeding_chunks = sum(
        count > MAX_TOKENS for count in token_sizes
    )

    output = {
        "document": document["document"],
        "chunking_configuration": {
            "strategy": "page_based_sentence_aware_token_aware_chunking",
            "model_name": MODEL_NAME,
            "model_token_limit": model_limit,
            "max_tokens": MAX_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS
        },
        "statistics": {
            "total_chunks": total_chunks,
            "average_tokens": round(average_tokens, 2),
            "minimum_tokens": min_tokens,
            "maximum_tokens": max_observed_tokens,
            "average_characters": round(average_characters, 2),
            "chunks_exceeding_limit": exceeding_chunks
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
    print(f"Average tokens: {average_tokens:.2f}")
    print(f"Minimum tokens: {min_tokens}")
    print(f"Maximum tokens: {max_observed_tokens}")
    print(f"Chunks exceeding limit: {exceeding_chunks}")

    print(f"\nSaved at: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_chunking()
