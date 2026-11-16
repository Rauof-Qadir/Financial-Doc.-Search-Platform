
from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_chunks.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_embeddings.npz"
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 8




def load_chunks(input_file: Path | str) -> list[dict]:
    """Load and validate document chunks from JSON."""

    input_file = Path(input_file)

    if not input_file.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {input_file}"
        )

    with input_file.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # Support both a JSON list and a dictionary containing "chunks".
    if isinstance(data, list):
        chunks = data
    elif isinstance(data, dict) and isinstance(data.get("chunks"), list):
        chunks = data["chunks"]
    else:
        raise ValueError(
            "Expected a list of chunks or a dictionary containing a chunks list."
        )

    if not chunks:
        raise ValueError("No document chunks found.")

    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk at index {index} must be a dictionary."
            )

        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(
                f"Chunk at index {index} contains empty or invalid text."
            )

        if not chunk.get("chunk_id"):
            raise ValueError(
                f"Chunk at index {index} is missing its chunk_id."
            )

    print(f"Loaded {len(chunks)} chunks successfully.")

    return chunks


def generate_embeddings(chunks: list[dict]) -> np.ndarray:
    """Generate normalized embeddings using a pretrained model."""

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu"
    )

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype(np.float32)


def save_embeddings(
    chunks: list[dict],
    embeddings: np.ndarray,
    output_file: Path
) -> None:
    """Save vectors and their corresponding document metadata."""

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks and embeddings must match."
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = [
        {
            "chunk_id": chunk["chunk_id"],
            "source_file": chunk["source_file"],
            "page_number": chunk["page_number"],
            "chunk_index": chunk["chunk_index"],
            "text": chunk["text"]
        }
        for chunk in chunks
    ]

    np.savez_compressed(
        output_file,
        embeddings=embeddings,
        metadata=np.array(
            json.dumps(metadata),
            dtype=np.str_
        )
    )

    print(f"Embeddings saved at: {output_file}")


def main():
    print("\nStarting embedding generation...\n")

    chunks = load_chunks(INPUT_FILE)

    embeddings = generate_embeddings(chunks)

    save_embeddings(
        chunks,
        embeddings,
        OUTPUT_FILE
    )

    print("\nEmbedding generation completed!")
    print(f"Total chunks: {len(chunks)}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Data type: {embeddings.dtype}")
    print(
        f"Average vector norm: "
        f"{np.linalg.norm(embeddings, axis=1).mean():.4f}"
    )


if __name__ == "__main__":
    main()
