
def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5
) -> float:
    """Calculate the proportion of top-k results that are relevant."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant IDs cannot be empty.")

    top_results = retrieved_ids[:k]

    hits = sum(
        1 for chunk_id in top_results
        if chunk_id in relevant
    )

    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5
) -> float:
    """Calculate the proportion of known relevant chunks retrieved."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("Relevant IDs cannot be empty.")

    retrieved_relevant = set(retrieved_ids[:k]) & relevant

    return len(retrieved_relevant) / len(relevant)


def reciprocal_rank(
    retrieved_ids: list[str],
    relevant_ids: list[str]
) -> float:
    """Return the reciprocal rank of the first relevant result."""

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("Relevant IDs cannot be empty.")

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int = 5
) -> float:
    """Return 1 if at least one relevant result is in top-k."""

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    relevant = set(relevant_ids)

    if not relevant:
        raise ValueError("Relevant IDs cannot be empty.")

    return float(
        any(chunk_id in relevant for chunk_id in retrieved_ids[:k])
    )
