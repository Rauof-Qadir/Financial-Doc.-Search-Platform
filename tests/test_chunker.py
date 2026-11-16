import re
import pytest

from src.preprocessing.chunker import (
    chunk_text,
    chunk_document,
    clean_page_text
)


class FakeTokenizer:
    """Lightweight tokenizer for unit tests."""

    def encode(
        self,
        text,
        add_special_tokens=True,
        truncation=False
    ):
        tokens = re.findall(r"\w+|[^\w\s]", text)

        if add_special_tokens:
            tokens = ["[CLS]"] + tokens + ["[SEP]"]

        return list(range(len(tokens)))


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def test_empty_text_returns_no_chunks(tokenizer):
    assert chunk_text("", tokenizer) == []


def test_short_text_returns_one_chunk(tokenizer):
    text = "Apple financial report"

    chunks = chunk_text(text, tokenizer)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_long_text_creates_multiple_chunks(tokenizer):
    text = "financial information " * 200

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=50,
        overlap=10
    )

    assert len(chunks) > 1


def test_invalid_chunk_size(tokenizer):
    with pytest.raises(ValueError):
        chunk_text(
            "financial report",
            tokenizer,
            max_tokens=0
        )


def test_invalid_overlap(tokenizer):
    with pytest.raises(ValueError):
        chunk_text(
            "financial report",
            tokenizer,
            max_tokens=100,
            overlap=100
        )


def test_missing_tokenizer():
    with pytest.raises(ValueError):
        chunk_text("financial report", None)


def test_document_metadata_is_preserved(tokenizer):
    pages = [
        {
            "page_number": 5,
            "text": "Apple financial report " * 100
        }
    ]

    chunks = chunk_document(
        pages,
        source_file="apple_annual_report.pdf",
        tokenizer=tokenizer,
        max_tokens=50,
        overlap=10
    )

    assert len(chunks) > 0
    assert chunks[0]["source_file"] == "apple_annual_report.pdf"
    assert chunks[0]["page_number"] == 5
    assert "chunk_id" in chunks[0]
    assert "token_count" in chunks[0]


def test_line_breaks_inside_sentence_are_normalized(tokenizer):
    text = (
        "Mac\n"
        "Mac net sales decreased 27% or $10.8 billion during "
        "2023 compared to 2022 due primarily to lower net "
        "sales of laptops.\n"
    )

    chunks = chunk_text(text, tokenizer)

    assert len(chunks) == 1

    assert (
        "Mac net sales decreased 27% or $10.8 billion during "
        "2023 compared to 2022 due primarily to lower net "
        "sales of laptops."
    ) in chunks[0]


def test_complete_sentences_are_preserved(tokenizer):
    first_sentence = (
        "Apple reported strong financial performance in 2023."
    )

    second_sentence = (
        "Services revenue increased during the same year."
    )

    text = f"{first_sentence} {second_sentence}"

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=11,
        overlap=0
    )

    assert len(chunks) == 2
    assert first_sentence in chunks[0]
    assert second_sentence in chunks[1]


def test_long_text_without_sentence_punctuation_is_split(tokenizer):
    text = "financial information " * 100

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=20,
        overlap=5
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= 20


def test_sentence_overlap_preserves_complete_units(tokenizer):
    text = (
        "Apple reported revenue growth in 2023. "
        "Services revenue increased by nine percent. "
        "Research and development expenses also increased."
    )

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=14,
        overlap=8
    )

    assert len(chunks) >= 2

    shared_words = set(chunks[0].split()) & set(
        chunks[1].split()
    )

    assert shared_words


def test_repeated_page_header_is_removed():
    text = (
        "Apple Inc.\n"
        "CONSOLIDATED BALANCE SHEETS\n"
        "Cash and cash equivalents\n"
        "29,965"
    )

    cleaned = clean_page_text(text)

    assert not cleaned.startswith("Apple Inc.")
    assert "CONSOLIDATED BALANCE SHEETS" in cleaned
    assert "29,965" in cleaned


def test_repeated_page_footer_is_removed():
    text = (
        "Total shareholders' equity\n"
        "62,146\n"
        "See accompanying Notes to Consolidated Financial Statements.\n"
        "Apple Inc. | 2023 Form 10-K | 30"
    )

    cleaned = clean_page_text(text)

    assert "62,146" in cleaned
    assert "See accompanying Notes" not in cleaned
    assert "Apple Inc. | 2023 Form 10-K" not in cleaned


def test_financial_data_and_note_references_are_preserved():
    text = (
        "Note 4 - Financial Instruments\n"
        "Cash and marketable securities\n"
        "$29,965\n"
        "(1)\n"
        "(2)\n"
        "Apple Inc. | 2023 Form 10-K | 36"
    )

    cleaned = clean_page_text(text)

    assert "29,965" in cleaned
    assert "(1)" in cleaned
    assert "(2)" in cleaned
    assert "Financial Instruments" in cleaned
    assert "Apple Inc. | 2023 Form 10-K" not in cleaned


def test_all_chunks_respect_token_limit(tokenizer):
    text = "Apple financial information increased during 2023. " * 30

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=30,
        overlap=8
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(tokenizer.encode(chunk)) <= 30


def test_all_original_words_are_preserved_without_overlap(tokenizer):
    text = "financial information " * 100

    chunks = chunk_text(
        text,
        tokenizer,
        max_tokens=30,
        overlap=0
    )

    original_words = text.split()

    chunk_words = [
        word
        for chunk in chunks
        for word in chunk.split()
    ]

    assert chunk_words == original_words

def test_long_sentence_rebalances_tiny_final_fragment():
    from src.preprocessing.chunker import chunk_text

    tokenizer = FakeTokenizer()

    # 228 words force a split, with a very small final fragment
    text = " ".join(f"word{i}" for i in range(228))

    chunks = chunk_text(
        text,
        tokenizer=tokenizer,
        max_tokens=220,
        overlap=0,
    )

    assert len(chunks) == 2

    token_counts = [
        len(tokenizer.encode(chunk, add_special_tokens=True))
        for chunk in chunks
    ]

    # Both fragments should be reasonably sized after rebalancing
    assert min(token_counts) >= 73

    # Neither fragment may exceed the model's chunk limit
    assert max(token_counts) <= 220




def test_repeated_trailing_footnote_markers_are_removed():
    from src.preprocessing.chunker import clean_page_text

    text = "Important financial statement. (1) (1) (1) (1) (2)"

    cleaned = clean_page_text(text)

    assert cleaned == "Important financial statement."


def test_single_trailing_footnote_marker_is_preserved():
    from src.preprocessing.chunker import clean_page_text

    text = "Exhibit reference 8-K 4.1 (1)"

    cleaned = clean_page_text(text)

    assert cleaned == text