
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSSearchEngine:

    def __init__(
        self,
        index_path="vector_store/apple_faiss.index",
        metadata_path="vector_store/apple_faiss_metadata.json",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(self.index_path)

        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)

        self.index = faiss.read_index(str(self.index_path))

        self.metadata = json.loads(
            self.metadata_path.read_text(encoding="utf-8")
        )

        if self.index.ntotal != len(self.metadata):
            raise ValueError("Index and metadata counts do not match.")

        self.model = SentenceTransformer(
            model_name,
            device="cpu",
        )

        print("FAISS Search Engine ready!")
        print("Indexed chunks:", self.index.ntotal)

    def search(self, query: str, top_k: int = 5):

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")

        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, idx in zip(scores[0], indices[0]):

            if idx == -1:
                continue

            chunk = self.metadata[int(idx)]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "page_number": chunk["page_number"],
                "score": float(score),
                "text": chunk["text"],
            })

        return results


if __name__ == "__main__":

    engine = FAISSSearchEngine()

    query = "What were Apple's research and development expenses in 2023?"

    results = engine.search(query, top_k=5)

    print("\nSEARCH QUERY:")
    print(query)

    print("\nTOP RETRIEVED RESULTS")
    print("=" * 70)

    for rank, result in enumerate(results, start=1):

        print(f"\nRANK: {rank}")
        print("CHUNK:", result["chunk_id"])
        print("PAGE:", result["page_number"])
        print("SCORE:", round(result["score"], 4))
        print("TEXT:", result["text"][:700])
        print("-" * 70)
