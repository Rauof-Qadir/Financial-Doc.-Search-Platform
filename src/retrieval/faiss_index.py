
import json
from pathlib import Path

import faiss
import numpy as np


def build_faiss_index(
    vector_store_path: str,
    index_path: str,
    metadata_path: str,
) -> None:

    vector_store = np.load(vector_store_path, allow_pickle=False)

    embeddings = vector_store["embeddings"].astype(
        np.float32,
        copy=True,
    )

    metadata_raw = vector_store["metadata"].item()
    metadata = json.loads(metadata_raw)

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array.")

    if len(embeddings) != len(metadata):
        raise ValueError(
            f"Count mismatch: {len(embeddings)} embeddings, "
            f"{len(metadata)} metadata records."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain invalid values.")

    if len({item["chunk_id"] for item in metadata}) != len(metadata):
        raise ValueError("Duplicate chunk IDs found.")

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    index_file = Path(index_path)
    metadata_file = Path(metadata_path)

    index_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(index_file))

    metadata_file.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("FAISS index created successfully!")
    print("Total vectors:", index.ntotal)
    print("Vector dimension:", dimension)
    print("Metadata records:", len(metadata))
    print("Index type: IndexFlatIP")
    print("Index saved:", index_file)
    print("Metadata saved:", metadata_file)


if __name__ == "__main__":
    build_faiss_index(
        vector_store_path=(
            "data/processed/"
            "apple_annual_report_token_aware_embeddings.npz"
        ),
        index_path="vector_store/apple_faiss.index",
        metadata_path="vector_store/apple_faiss_metadata.json",
    )
