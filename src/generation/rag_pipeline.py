
from typing import Any

from src.generation.context_builder import build_context
from src.retrieval.bm25_search import BM25SearchEngine


class FinancialRAGPipeline:
    """
    Connect BM25 retrieval with evidence context construction.

    This component retrieves supporting financial document chunks.
    It does not generate a natural-language answer yet.
    """

    def __init__(
        self,
        metadata: list[dict[str, Any]],
        top_k: int = 5,
    ):
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        self.retriever = BM25SearchEngine(metadata)
        self.top_k = top_k

    def run(self, query: str) -> dict[str, Any]:
        """
        Retrieve relevant chunks and build their evidence context.

        Returns the original query, retrieved chunks,
        formatted context, and evidence availability.
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        retrieved_chunks = self.retriever.search(
            query=query,
            top_k=self.top_k,
        )

        context = build_context(
            results=retrieved_chunks,
            max_chunks=self.top_k,
        )

        return {
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "context": context,
            "evidence_found": bool(retrieved_chunks),
        }
