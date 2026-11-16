
from pathlib import Path
import json

from src.retrieval.semantic_search import SemanticSearchEngine
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_eval_dataset.json"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "retrieval_evaluation.json"
)

TOP_K = 5


def load_evaluation_dataset(file_path: Path) -> list[dict]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty list.")

    for item in dataset:
        if not isinstance(item, dict):
            raise ValueError("Each evaluation item must be an object.")

        if not item.get("question"):
            raise ValueError("Evaluation question cannot be empty.")

        relevant_ids = item.get("relevant_chunk_ids")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError("Each question needs relevant chunk IDs.")

    return dataset


def evaluate_search(engine, dataset: list[dict]) -> dict:
    question_results = []

    for item in dataset:
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]

        print(f"\nEvaluating: {question}")

        retrieved = engine.search(
            query=question,
            top_k=TOP_K
        )

        retrieved_ids = [
            result["chunk_id"] for result in retrieved
        ]

        metrics = {
            "precision_at_5": precision_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
            "recall_at_5": recall_at_k(
                retrieved_ids, relevant_ids, TOP_K
            ),
            "mrr": reciprocal_rank(
                retrieved_ids, relevant_ids
            ),
            "hit_rate_at_5": hit_rate_at_k(
                retrieved_ids, relevant_ids, TOP_K
            )
        }

        question_results.append({
            "question_id": item.get("question_id"),
            "question": question,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids,
            "metrics": metrics
        })

        print(
            f"Precision@5: {metrics['precision_at_5']:.4f}"
        )
        print(
            f"Recall@5:    {metrics['recall_at_5']:.4f}"
        )
        print(
            f"MRR:         {metrics['mrr']:.4f}"
        )
        print(
            f"Hit Rate@5:  {metrics['hit_rate_at_5']:.4f}"
        )

    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "hit_rate_at_5"
    ]

    averages = {
        metric: sum(
            result["metrics"][metric]
            for result in question_results
        ) / len(question_results)
        for metric in metric_names
    }

    return {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "top_k": TOP_K,
        "total_questions": len(dataset),
        "average_metrics": averages,
        "question_results": question_results
    }


def main():
    print("\nFinancial Document Retrieval Evaluation")
    print("=" * 48)

    dataset = load_evaluation_dataset(DATASET_FILE)

    print(f"Evaluation questions: {len(dataset)}")

    engine = SemanticSearchEngine()

    report = evaluate_search(engine, dataset)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nOverall Evaluation Results")
    print("-" * 48)

    for metric, score in report["average_metrics"].items():
        print(f"{metric}: {score:.4f}")

    print(f"\nReport saved at: {REPORT_FILE}")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
