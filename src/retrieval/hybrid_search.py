
from pathlib import Path
import re

import numpy as np
from rank_bm25 import BM25Okapi

from src.retrieval.semantic_search import (
    SemanticSearchEngine,
    VECTOR_STORE_FILE,
)


def tokenize(text: str) -> list[str]:
    """Convert text into lowercase word tokens for BM25."""

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    rrf_k: int = 60,
) -> list[dict]:
    """
    Combine multiple ranked result lists using Reciprocal Rank Fusion.

    Each result must contain a unique chunk_id.
    Ranks start at 1.
    """

    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than zero.")

    scores = {}
    documents = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                raise ValueError("Every result must contain a chunk_id.")

            scores[chunk_id] = scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )

            # Keep the most complete result metadata.
            documents[chunk_id] = {
                **documents.get(chunk_id, {}),
                **item,
            }

    fused_results = []

    for chunk_id, score in scores.items():
        result = documents[chunk_id].copy()
        result["rrf_score"] = float(score)
        fused_results.append(result)

    fused_results.sort(
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return fused_results


class HybridSearchEngine:
    """Combine semantic search and BM25 keyword search."""


    def __init__(
        self,
        vector_store_file: str | Path = VECTOR_STORE_FILE,
        semantic_engine=None,
    ):
        print("Initializing hybrid search...")

        # Reuse an existing engine when supplied.
        # Otherwise, create a new semantic search engine.
        if semantic_engine is None:
            self.semantic_engine = SemanticSearchEngine(
                vector_store_file=vector_store_file
            )
        else:
            self.semantic_engine = semantic_engine

        self.metadata = self.semantic_engine.metadata

        if not self.metadata:
            raise ValueError("Document collection cannot be empty.")

        chunk_ids = [
            item.get("chunk_id")
            for item in self.metadata
        ]

        if any(not chunk_id for chunk_id in chunk_ids):
            raise ValueError("Every document chunk must have a chunk_id.")

        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("Document chunk IDs must be unique.")

        print("Building BM25 keyword index...")

        tokenized_corpus = [
            tokenize(item.get("text", ""))
            for item in self.metadata
        ]

        if any(not tokens for tokens in tokenized_corpus):
            raise ValueError(
                "Every document chunk must contain searchable text."
            )

        self.bm25 = BM25Okapi(tokenized_corpus)

        print(
            f"Hybrid search ready: {len(self.metadata)} chunks loaded."
        )


    def search(
        self,
        query: str,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict]:
        """Retrieve and fuse semantic and keyword search results."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        query = query.strip()
        query_tokens = tokenize(query)

        if not query_tokens:
            raise ValueError("Query must contain searchable words or numbers.")

        # 1. Semantic search: rank every chunk by meaning.
        semantic_results = self.semantic_engine.search(
            query,
            top_k=len(self.metadata),
        )

        for rank, result in enumerate(semantic_results, start=1):
            result["semantic_rank"] = rank

        # 2. BM25 search: rank chunks by keyword relevance.
        bm25_scores = self.bm25.get_scores(query_tokens)

        keyword_indices = np.argsort(-bm25_scores)

        keyword_results = []

        for index in keyword_indices:
            score = float(bm25_scores[index])

            # Ignore zero-score documents. They have no keyword match.
            if score <= 0:
                continue

            result = self.metadata[int(index)].copy()
            result["bm25_score"] = score
            result["bm25_rank"] = len(keyword_results) + 1

            keyword_results.append(result)

        # 3. Fuse both ranked lists.
        fused_results = reciprocal_rank_fusion(
            rankings=[
                semantic_results,
                keyword_results,
            ],
            rrf_k=rrf_k,
        )

        return fused_results[:top_k]


def main():
    print("\nFinancial Document Hybrid Search")
    print("=" * 50)

    engine = HybridSearchEngine()

    print("\nHybrid search is ready!")
    print("Type 'exit' to stop.\n")

    while True:
        query = input("Enter your question: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Search session ended.")
            break

        if not query:
            print("Please enter a question.\n")
            continue

        try:
            results = engine.search(query, top_k=5)

            print("\nTop hybrid search results")
            print("-" * 50)

            for rank, result in enumerate(results, start=1):
                print(f"\nResult #{rank}")
                print(f"RRF Score: {result['rrf_score']:.5f}")
                print(
                    f"Semantic Rank: "
                    f"{result.get('semantic_rank', 'Not ranked')}"
                )
                print(
                    f"BM25 Rank: "
                    f"{result.get('bm25_rank', 'Not ranked')}"
                )
                print(f"Source: {result['source_file']}")
                print(f"Page: {result['page_number']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Text: {result['text'][:700]}")

            print("\n" + "=" * 50 + "\n")

        except ValueError as error:
            print(f"Invalid search: {error}\n")


if __name__ == "__main__":
    main()
