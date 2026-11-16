
from pathlib import Path
import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_chunks.json"
)

DEFAULT_OUTPUT_FILE = (
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

    seen_ids = set()

    for index, chunk in enumerate(chunks):

        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk at index {index} must be a dictionary."
            )

        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise ValueError(
                f"Chunk at index {index} contains empty or invalid text."
            )

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            raise ValueError(
                f"Chunk at index {index} is missing its chunk_id."
            )

        if chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate chunk ID found: {chunk_id}"
            )

        seen_ids.add(chunk_id)

        required_fields = [
            "source_file",
            "page_number",
            "chunk_index",
        ]

        for field in required_fields:
            if field not in chunk:
                raise ValueError(
                    f"Chunk {chunk_id} is missing field: {field}"
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

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32
    )

    if embeddings.ndim != 2:
        raise ValueError("Generated embeddings must be a 2D array.")

    if len(embeddings) != len(chunks):
        raise ValueError(
            "Generated embedding count does not match chunk count."
        )

    if not np.isfinite(embeddings).all():
        raise ValueError(
            "Generated embeddings contain invalid values."
        )

    return embeddings


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

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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
    parser = argparse.ArgumentParser(
        description="Generate embeddings for financial document chunks."
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help="Path to the input chunks JSON file."
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Path where the embeddings NPZ file will be saved."
    )

    args = parser.parse_args()

    print("\nStarting embedding generation...\n")

    chunks = load_chunks(args.input_file)

    embeddings = generate_embeddings(chunks)

    save_embeddings(
        chunks,
        embeddings,
        args.output_file
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
