import json
from pathlib import Path

REPORT_PATH = Path("reports/retrieval_comparison.json")
DATASET_PATH = Path("evaluation/retrieval_eval_dataset.json")
CHUNKS_PATH = Path("data/processed/apple_annual_report_chunks.json")

FAILED_IDS = ["Q005", "Q007", "Q013", "Q014", "Q022", "Q023"]

# Load benchmark report
report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))

# Load evaluation questions and gold labels
dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

questions = {
    item["question_id"]: item
    for item in dataset
}

# Load document chunks
chunks_data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

chunks = {
    chunk["chunk_id"]: chunk
    for chunk in chunks_data["chunks"]
}

SYSTEMS = ["semantic", "bm25", "hybrid"]

for question_id in FAILED_IDS:

    if question_id not in questions:
        print(f"\nWARNING: {question_id} not found in evaluation dataset.")
        continue

    question = questions[question_id]
    gold_ids = set(question["relevant_chunk_ids"])

    print("\n" + "=" * 100)
    print(f"QUESTION ID: {question_id}")
    print("=" * 100)

    print("\nQUESTION:")
    print(question["question"])

    print("\nEXPECTED RELEVANT CHUNKS (GOLD LABELS):")

    for chunk_id in sorted(gold_ids):
        print(f"  - {chunk_id}")

    for system in SYSTEMS:

        results = report["systems"][system]["question_results"]

        result = next(
            (item for item in results
             if item["question_id"] == question_id),
            None
        )

        if result is None:
            print(f"\n{system.upper()}: No result found.")
            continue

        retrieved_ids = result["retrieved_chunk_ids"]
        metrics = result["metrics"]

        print("\n" + "-" * 100)
        print(f"{system.upper()} SEARCH")
        print("-" * 100)

        print(
            f"Precision@5: {metrics['precision_at_5']:.4f} | "
            f"Recall@5: {metrics['recall_at_5']:.4f} | "
            f"MRR: {metrics['mrr']:.4f} | "
            f"Hit Rate@5: {metrics['hit_rate_at_5']:.4f}"
        )

        relevant_found = 0

        for rank, chunk_id in enumerate(retrieved_ids, start=1):

            is_relevant = chunk_id in gold_ids

            if is_relevant:
                relevant_found += 1
                status = "YES"
            else:
                status = "NO"

            chunk = chunks.get(chunk_id)

            print(
                f"\nRank: {rank} | Relevant: {status}"
            )
            print(f"Chunk ID: {chunk_id}")

            if chunk:
                print(f"Page: {chunk['page_number']}")
                print("Text:")
                print(chunk["text"][:600].replace("\n", " "))
            else:
                print("WARNING: Chunk text not found.")

        missing_ids = gold_ids - set(retrieved_ids)

        print("\nSYSTEM SUMMARY:")
        print(f"Relevant chunks retrieved: {relevant_found}/{len(gold_ids)}")

        if missing_ids:
            print("Missing gold chunks:")
            for chunk_id in sorted(missing_ids):
                print(f"  - {chunk_id}")
        else:
            print("All gold chunks were retrieved.")

print("\n" + "=" * 100)
print("RETRIEVAL DIAGNOSTICS COMPLETED")
print("=" * 100)

print(
    "\nNOTE: Retrieval scores are not stored in the current benchmark report. "
    "Only ranking positions and relevance are available."
)
