import re
from pathlib import Path



def clean_page_text(text: str) -> str:
    """Remove repeated report headers and footers."""

    text = re.sub(
        r"^Apple Inc\.\s*\n",
        "",
        text.strip(),
        count=1
    )

    text = re.sub(
        r"\s*See accompanying Notes to Consolidated Financial Statements\.\s*"
        r"Apple Inc\. \| 2023 Form 10-K \| \d+\s*$",
        "",
        text
    )

    text = re.sub(
        r"\s*Apple Inc\. \| 2023 Form 10-K \| \d+\s*$",
        "",
        text
    )

    text = re.sub(
    r"(?:\s*\(1\)){4}\s*\(2\)\s*$",
    "",
    text
)

    return text.strip()



def _normalize_text(text: str) -> str:
    """Normalize whitespace while retaining readable text."""

    return re.sub(r"\s+", " ", text).strip()


def _split_sentences(text: str) -> list[str]:
    """Split text at likely sentence boundaries."""

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


def _token_count(text: str, tokenizer) -> int:
    """Count actual tokens without padding or truncation."""

    backend = getattr(tokenizer, "backend_tokenizer", None)

    if backend is not None:
        # Prevent inherited padding/truncation settings from corrupting counts.
        backend.no_padding()
        backend.no_truncation()

        encoding = backend.encode(
            text,
            add_special_tokens=True
        )
        return len(encoding.ids)

    if hasattr(tokenizer, "tokenize"):
        token_ids = tokenizer.tokenize(text)
        special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
        return len(token_ids) + special_tokens

    return len(
        tokenizer.encode(
            text,
            add_special_tokens=True,
            truncation=False
        )
    )



def _split_long_unit(
    text: str,
    tokenizer,
    max_tokens: int
) -> list[str]:
    """
    Split oversized text at word boundaries and rebalance
    a very small final fragment with the preceding piece.
    """
    words = text.split()

    if not words:
        return []

    pieces = []
    current_words = []

    for word in words:
        candidate_words = current_words + [word]
        candidate = " ".join(candidate_words)

        if _token_count(candidate, tokenizer) <= max_tokens:
            current_words = candidate_words
            continue

        if current_words:
            pieces.append(" ".join(current_words))

        if _token_count(word, tokenizer) > max_tokens:
            raise ValueError(
                f"A single word exceeds the token limit: {word!r}"
            )

        current_words = [word]

    if current_words:
        pieces.append(" ".join(current_words))

    # Rebalance a very small final piece.
    if len(pieces) >= 2:
        previous_words = pieces[-2].split()
        final_words = pieces[-1].split()

        minimum_target = max_tokens // 3

        while len(previous_words) > 1:
            final_text = " ".join(final_words)

            if _token_count(final_text, tokenizer) >= minimum_target:
                break

            candidate_previous = previous_words[:-1]
            candidate_final = [previous_words[-1]] + final_words

            previous_text = " ".join(candidate_previous)
            final_text = " ".join(candidate_final)

            if _token_count(previous_text, tokenizer) < minimum_target:
                break

            if _token_count(final_text, tokenizer) > max_tokens:
                break

            previous_words = candidate_previous
            final_words = candidate_final

        pieces[-2] = " ".join(previous_words)
        pieces[-1] = " ".join(final_words)

    return pieces


def _build_units(
    text: str,
    tokenizer,
    max_tokens: int
) -> list[str]:
    """Create sentence-aware units within the token limit."""

    sentences = _split_sentences(text)

    units = []

    for sentence in sentences:

        if _token_count(sentence, tokenizer) <= max_tokens:
            units.append(sentence)

        else:
            units.extend(
                _split_long_unit(
                    sentence,
                    tokenizer,
                    max_tokens
                )
            )

    return units


def chunk_text(
    text: str,
    tokenizer,
    max_tokens: int = 220,
    overlap: int = 40
) -> list[str]:
    """
    Create token-aware overlapping chunks.

    max_tokens includes tokenizer special tokens.
    overlap is an approximate token target.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive.")

    if overlap < 0 or overlap >= max_tokens:
        raise ValueError(
            "overlap must be non-negative and smaller than max_tokens."
        )

    if tokenizer is None:
        raise ValueError("A tokenizer must be provided.")

    text = _normalize_text(text)

    if not text:
        return []

    units = _build_units(
        text,
        tokenizer,
        max_tokens
    )

    if not units:
        return []

    chunks = []
    start = 0

    while start < len(units):

        current_units = []
        end = start

        while end < len(units):

            candidate_units = current_units + [units[end]]
            candidate = " ".join(candidate_units)

            if (
                current_units
                and _token_count(candidate, tokenizer) > max_tokens
            ):
                break

            if _token_count(candidate, tokenizer) > max_tokens:
                raise ValueError(
                    "An individual text unit exceeds max_tokens."
                )

            current_units.append(units[end])
            end += 1

        chunk = " ".join(current_units).strip()

        if chunk:
            actual_tokens = _token_count(chunk, tokenizer)

            if actual_tokens > max_tokens:
                raise ValueError(
                    f"Generated chunk has {actual_tokens} tokens; "
                    f"maximum is {max_tokens}."
                )

            chunks.append(chunk)

        if end >= len(units):
            break

        # Find a complete-unit overlap close to the target.
        next_start = end
        overlap_tokens = 0

        for index in range(end - 1, start - 1, -1):

            candidate_units = (
                units[index:end]
            )

            candidate = " ".join(candidate_units)
            candidate_tokens = _token_count(candidate, tokenizer)

            if candidate_tokens > overlap:
                break

            overlap_tokens = candidate_tokens
            next_start = index

        # Always guarantee forward progress.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_document(
    pages: list[dict],
    source_file: str,
    tokenizer,
    max_tokens: int = 220,
    overlap: int = 40
) -> list[dict]:
    """Chunk document pages while preserving source metadata."""

    document_chunks = []

    for page in pages:

        page_number = page["page_number"]
        page_text = clean_page_text(page["text"])

        page_chunks = chunk_text(
            page_text,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
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
                "character_count": len(chunk),
                "token_count": _token_count(chunk, tokenizer)
            })

    return document_chunks
