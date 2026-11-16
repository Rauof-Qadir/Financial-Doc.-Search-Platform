
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.hybrid_search import tokenize


class BM25SearchEngine:
    """Search document chunks using BM25 keyword relevance."""

    def __init__(self, metadata: list[dict]):
        if not isinstance(metadata, list) or not metadata:
            raise ValueError("Metadata must be a non-empty list.")

        self.metadata = metadata

        chunk_ids = [
            item.get("chunk_id")
            for item in metadata
        ]

        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("Every chunk must contain a chunk_id.")

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Chunk IDs must be unique.")

        tokenized_corpus = [
            tokenize(item.get("text", ""))
            for item in metadata
        ]

        if any(not tokens for tokens in tokenized_corpus):
            raise ValueError("Every chunk must contain searchable text.")

        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Return top document chunks ranked by BM25."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError(
                "Query must contain searchable words or numbers."
            )

        scores = self.bm25.get_scores(query_tokens)
        indices = np.argsort(-scores)

        results = []

        for index in indices:
            score = float(scores[index])

            if score <= 0:
                continue

            result = self.metadata[int(index)].copy()
            result["bm25_score"] = score

            results.append(result)

            if len(results) == top_k:
                break

        return results
