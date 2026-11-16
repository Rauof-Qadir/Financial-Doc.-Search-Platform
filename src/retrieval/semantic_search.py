
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VECTOR_STORE_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_embeddings.npz"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"



def load_vector_store(vector_store_file: str | Path):
    """Load document vectors and their metadata."""

    vector_store_file = Path(vector_store_file)

    if not vector_store_file.exists():
        raise FileNotFoundError(
            f"Vector store not found: {vector_store_file}"
        )

    with np.load(vector_store_file, allow_pickle=False) as data:
        if "embeddings" not in data or "metadata" not in data:
            raise ValueError("Vector store is missing required arrays.")

        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        metadata_json = str(data["metadata"].item())

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Embeddings must be a non-empty 2D array.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid numerical values.")

    metadata = json.loads(metadata_json)

    if not isinstance(metadata, list):
        raise ValueError("Metadata must be a list.")

    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embeddings count do not match.")

    return embeddings, metadata



def rank_chunks(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    metadata: list[dict],
    top_k: int = 5
) -> list[dict]:
    """Rank document chunks using cosine similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    ).reshape(-1)

    embeddings = np.asarray(embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Document embeddings must be a 2D array.")

    if query_embedding.shape[0] != embeddings.shape[1]:
        raise ValueError("Query and document dimensions do not match.")

    if len(metadata) != len(embeddings):
        raise ValueError("Metadata and embeddings count do not match.")

    if not np.isfinite(query_embedding).all():
        raise ValueError("Query embedding contains invalid values.")

    query_norm = np.linalg.norm(query_embedding)

    if query_norm == 0:
        raise ValueError("Query embedding cannot be a zero vector.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Document embeddings contain invalid values.")

    document_norms = np.linalg.norm(embeddings, axis=1)

    if np.any(document_norms == 0):
        raise ValueError("Document embeddings cannot contain zero vectors.")

    # Normalize vectors so dot product equals cosine similarity.
    normalized_query = query_embedding / query_norm
    normalized_documents = embeddings / document_norms[:, np.newaxis]

    similarities = normalized_documents @ normalized_query

    # Select indices of the highest similarity scores.
    top_indices = np.argsort(-similarities)[:top_k]

    results = []

    for index in top_indices:
        result = metadata[index].copy()
        result["similarity_score"] = float(similarities[index])
        results.append(result)

    return results


class SemanticSearchEngine:
    """Load the embedding model and search the document collection."""

    def __init__(
        self,
        vector_store_file: Path = VECTOR_STORE_FILE
    ):
        print("Loading document vector store...")

        self.embeddings, self.metadata = load_vector_store(
            vector_store_file
        )

        print(f"Loaded {len(self.metadata)} document chunks.")

        print(f"Loading embedding model: {MODEL_NAME}")

        self.model = SentenceTransformer(
            MODEL_NAME,
            device="cpu"
        )

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict]:
        """Convert a question into a vector and retrieve relevant chunks."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        query_embedding = self.model.encode(
            query.strip(),
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return rank_chunks(
            query_embedding=query_embedding,
            embeddings=self.embeddings,
            metadata=self.metadata,
            top_k=top_k
        )


def main():
    print("\nFinancial Document Semantic Search")
    print("=" * 45)

    engine = SemanticSearchEngine()

    print("\nSearch is ready!")
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

            print("\nTop relevant passages")
            print("-" * 45)

            for rank, result in enumerate(results, start=1):
                print(f"\nResult #{rank}")
                print(f"Similarity: {result['similarity_score']:.4f}")
                print(f"Source: {result['source_file']}")
                print(f"Page: {result['page_number']}")
                print(f"Chunk ID: {result['chunk_id']}")
                print(f"Text: {result['text'][:700]}")

            print("\n" + "=" * 45 + "\n")

        except ValueError as error:
            print(f"Invalid search: {error}\n")


if __name__ == "__main__":
    main()
