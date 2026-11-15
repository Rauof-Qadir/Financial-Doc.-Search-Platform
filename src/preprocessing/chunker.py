
from pathlib import Path


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[str]:
    """
    Split text into overlapping chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        # Prefer ending at a word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(end - overlap, start + 1)

        # Avoid starting in the middle of whitespace
        while next_start < len(text) and text[next_start].isspace():
            next_start += 1

        start = next_start

    return chunks


def chunk_document(
    pages: list[dict],
    source_file: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[dict]:
    """
    Chunk document pages while preserving source metadata.
    """

    document_chunks = []

    for page in pages:

        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(page_chunks, start=1):

            document_chunks.append({
                "chunk_id": (
                    f"{Path(source_file).stem}"
                    f"-p{page_number:03d}"
                    f"-c{index:03d}"
                ),
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": index,
                "text": chunk,
                "character_count": len(chunk)
            })

    return document_chunks
