
import re
from pathlib import Path


def clean_page_text(text: str) -> str:
    """
    Remove known repeated Apple annual-report page headers
    and footers without removing financial information.
    """

    # Remove the repeated company header only at page start.
    text = re.sub(
        r"^Apple Inc\.\s*\n",
        "",
        text.strip(),
        count=1
    )

    # Remove the repeated financial-statement note footer
    # and the printed page-number footer at page end.
    text = re.sub(
        r"\s*See accompanying Notes to Consolidated Financial Statements\.\s*"
        r"Apple Inc\. \| 2023 Form 10-K \| \d+\s*$",
        "",
        text
    )

    # Remove a standalone printed page footer if present.
    text = re.sub(
        r"\s*Apple Inc\. \| 2023 Form 10-K \| \d+\s*$",
        "",
        text
    )

    return text.strip()

def _normalize_text(text: str) -> str:
    """
    Replace line breaks and repeated whitespace with single spaces.

    PDF extraction often inserts line breaks inside sentences.
    Normalizing whitespace prevents those line breaks from
    being treated as paragraph or sentence boundaries.
    """

    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """
    Split text at likely sentence boundaries.

    A boundary is recognized when ., !, or ? is followed by
    whitespace and then an uppercase letter, digit, or quote.
    """

    if not text:
        return []

    sentences = re.split(
        r'(?<=[.!?])\s+(?=[A-Z0-9“"(])',
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]



def _split_long_unit(
    text: str,
    chunk_size: int
) -> list[str]:
    """
    Split a long text unit at word boundaries.

    If the final piece is unusually small, rebalance the
    last two pieces to avoid tiny chunks containing useful
    financial values or table continuations.
    """

    words = text.split()

    if not words:
        return []

    pieces = []
    current_words = []
    current_length = 0

    # First, create chunks using the normal size limit.
    for word in words:

        additional_length = (
            len(word) if not current_words
            else len(word) + 1
        )

        if (
            current_words
            and current_length + additional_length > chunk_size
        ):
            pieces.append(" ".join(current_words))
            current_words = [word]
            current_length = len(word)

        else:
            current_words.append(word)
            current_length += additional_length

    if current_words:
        pieces.append(" ".join(current_words))

    # Avoid a very small final piece by rebalancing
    # the last two pieces.
    minimum_size = min(100, max(1, chunk_size // 4))

    if len(pieces) >= 2 and len(pieces[-1]) < minimum_size:

        combined_words = (
            pieces[-2].split() + pieces[-1].split()
        )

        best_split = None
        best_difference = float("inf")

        for split_index in range(1, len(combined_words)):

            left = " ".join(combined_words[:split_index])
            right = " ".join(combined_words[split_index:])

            if len(left) > chunk_size or len(right) > chunk_size:
                continue

            difference = abs(len(left) - len(right))

            if difference < best_difference:
                best_difference = difference
                best_split = (left, right)

        if best_split is not None:
            pieces[-2:] = list(best_split)

    return pieces



def _build_units(
    text: str,
    chunk_size: int
) -> list[str]:
    """
    Create sentence-aware units.

    Long sentences are split at word boundaries so that
    extremely long units do not create oversized chunks.
    """

    sentences = _split_sentences(text)

    units = []

    for sentence in sentences:

        if len(sentence) > chunk_size:
            units.extend(
                _split_long_unit(sentence, chunk_size)
            )

        else:
            units.append(sentence)

    return units


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200
) -> list[str]:
    """
    Split text into sentence-aware overlapping chunks.

    chunk_size:
        Target maximum chunk length in characters.

    overlap:
        Approximate overlap in characters. Overlap is created
        using complete sentence units whenever possible.

    A chunk can exceed chunk_size when a sentence itself is
    longer than the target and cannot be split further without
    breaking a word.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = _normalize_text(text)

    if not text:
        return []

    units = _build_units(
        text=text,
        chunk_size=chunk_size
    )

    if not units:
        return []

    chunks = []
    start = 0

    while start < len(units):

        current_units = []
        current_length = 0
        end = start

        # Add complete units until the target size is reached.
        while end < len(units):

            unit = units[end]

            additional_length = (
                len(unit) if not current_units
                else len(unit) + 1
            )

            if (
                current_units
                and current_length + additional_length > chunk_size
            ):
                break

            current_units.append(unit)
            current_length += additional_length
            end += 1

        chunk = " ".join(current_units).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(units):
            break

        # Find sentence-level overlap for the next chunk.
        next_start = end
        overlap_length = 0

        for index in range(end - 1, start - 1, -1):

            unit_length = len(units[index])

            additional_length = (
                unit_length if overlap_length == 0
                else unit_length + 1
            )

            if overlap_length + additional_length > overlap:
                break

            overlap_length += additional_length
            next_start = index

        # Guarantee forward progress, even if one unit is
        # larger than the requested overlap.
        if next_start <= start:
            next_start = end

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
        page_text = clean_page_text(page["text"])

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
