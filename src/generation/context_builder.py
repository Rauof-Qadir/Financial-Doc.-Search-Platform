
from typing import Any


def build_context(
    results: list[dict[str, Any]],
    max_chunks: int = 5,
) -> str:
    """
    Convert retrieved document chunks into structured evidence.

    Each result must contain:
    - chunk_id
    - page_number
    - text

    The resulting context can be passed to a future
    question-answering model with source references.
    """

    if not isinstance(results, list):
        raise TypeError("results must be a list.")

    if not isinstance(max_chunks, int) or isinstance(max_chunks, bool):
        raise ValueError("max_chunks must be a positive integer.")

    if max_chunks < 1:
        raise ValueError("max_chunks must be a positive integer.")

    if not results:
        return "No supporting evidence was retrieved."

    context_parts = []

    for index, result in enumerate(results[:max_chunks], start=1):

        if not isinstance(result, dict):
            raise TypeError("Each result must be a dictionary.")

        required_fields = ("chunk_id", "page_number", "text")

        for field in required_fields:
            if field not in result:
                raise ValueError(f"Missing required field: {field}")

        chunk_id = result["chunk_id"]
        page_number = result["page_number"]
        chunk_text = result["text"]

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")

        if not isinstance(page_number, int) or isinstance(page_number, bool):
            raise ValueError("page_number must be an integer.")

        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError("text must be a non-empty string.")

        context_parts.append(
            f"SOURCE {index}\n"
            f"Chunk ID: {chunk_id}\n"
            f"Page: {page_number}\n"
            f"Evidence:\n{chunk_text.strip()}"
        )

    return "\n\n".join(context_parts)
