from pathlib import Path
from typing import Any

from src.ingestion.pdf_loader import extract_pdf
from src.preprocessing.chunker import chunk_document


DEFAULT_MAX_TOKENS = 220
DEFAULT_OVERLAP_TOKENS = 40


def process_pdf(
    pdf_path: str | Path,
    tokenizer: Any,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap: int = DEFAULT_OVERLAP_TOKENS,
) -> dict[str, Any]:
    """
    Extract, clean, and chunk any supported text-based PDF.

    The tokenizer is injected by the application so that
    the expensive tokenizer initialization can be cached.

    Returns document information, chunks, and processing statistics.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    if tokenizer is None:
        raise ValueError("A tokenizer must be provided.")

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(
            "overlap must be non-negative and smaller than max_tokens."
        )

    # Step 1: Extract text and metadata from the PDF.
    extracted = extract_pdf(pdf_path)

    pages = extracted["pages"]
    full_text = extracted["full_text"]

    if not full_text.strip():
        raise ValueError(
            "This PDF contains no extractable text. "
            "It may be a scanned or image-only PDF. "
            "OCR support is required for this document."
        )

    # Step 2: Generate token-aware chunks.
    chunks = chunk_document(
        pages=pages,
        source_file=extracted["source_file"],
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap=overlap,
    )

    if not chunks:
        raise ValueError(
            "No searchable text chunks could be generated from this PDF."
        )

    # Step 3: Calculate document statistics.
    token_counts = [
        chunk["token_count"]
        for chunk in chunks
    ]

    character_counts = [
        chunk["character_count"]
        for chunk in chunks
    ]

    document_info = {
        "source_file": extracted["source_file"],
        "total_pages": extracted["total_pages"],
        "total_words": len(full_text.split()),
        "total_characters": len(full_text),
        "metadata": extracted["metadata"],
    }

    statistics = {
        "total_chunks": len(chunks),
        "average_tokens": round(
            sum(token_counts) / len(token_counts), 2
        ),
        "minimum_tokens": min(token_counts),
        "maximum_tokens": max(token_counts),
        "average_characters": round(
            sum(character_counts) / len(character_counts), 2
        ),
        "chunks_exceeding_limit": sum(
            count > max_tokens
            for count in token_counts
        ),
    }

    # Step 4: Return a reusable document package.
    return {
        "document": document_info,
        "chunking_configuration": {
            "strategy": "page_based_sentence_aware_token_aware_chunking",
            "max_tokens": max_tokens,
            "overlap_tokens": overlap,
        },
        "statistics": statistics,
        "chunks": chunks,
    }
