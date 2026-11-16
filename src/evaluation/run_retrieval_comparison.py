
from pathlib import Path
import json

from src.retrieval.semantic_search import SemanticSearchEngine
from src.retrieval.bm25_search import BM25SearchEngine
from src.retrieval.hybrid_search import HybridSearchEngine

from src.evaluation.run_retrieval_evaluation import (
    load_evaluation_dataset,
    evaluate_search,
    DATASET_FILE,
    TOP_K,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "retrieval_comparison.json"
)


def main():
    print("\nFinancial Document Retrieval Benchmark")
    print("=" * 55)

    dataset = load_evaluation_dataset(DATASET_FILE)

    print(f"Evaluation questions: {len(dataset)}")
    print(f"Top K: {TOP_K}")

    # Load the embedding model only once.
    semantic_engine = SemanticSearchEngine()

    # Initialize the three retrieval systems.
    bm25_engine = BM25SearchEngine(
        metadata=semantic_engine.metadata
    )

    hybrid_engine = HybridSearchEngine(
        semantic_engine=semantic_engine
    )

    engines = {
        "semantic": semantic_engine,
        "bm25": bm25_engine,
        "hybrid": hybrid_engine,
    }

    comparison = {
        "top_k": TOP_K,
        "total_questions": len(dataset),
        "systems": {},
    }

    # Evaluate each system on exactly the same dataset.
    for system_name, engine in engines.items():
        print("\n" + "=" * 55)
        print(f"EVALUATING: {system_name.upper()} SEARCH")
        print("=" * 55)

        report = evaluate_search(engine, dataset)

        comparison["systems"][system_name] = report

    # Compare the average metrics.
    metric_names = [
        "precision_at_5",
        "recall_at_5",
        "mrr",
        "hit_rate_at_5",
    ]

    comparison["summary"] = {}

    for metric in metric_names:
        comparison["summary"][metric] = {
            system_name: report["average_metrics"][metric]
            for system_name, report in comparison["systems"].items()
        }

    # Save report.
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_FILE.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    # Display comparison table.
    print("\n\nFINAL RETRIEVAL COMPARISON")
    print("=" * 75)

    print(
        f"{'Metric':<20}"
        f"{'Semantic':>15}"
        f"{'BM25':>15}"
        f"{'Hybrid':>15}"
    )

    print("-" * 75)

    for metric, scores in comparison["summary"].items():
        print(
            f"{metric:<20}"
            f"{scores['semantic']:>15.4f}"
            f"{scores['bm25']:>15.4f}"
            f"{scores['hybrid']:>15.4f}"
        )

    print("\nBenchmark report saved at:")
    print(REPORT_FILE)

    print("\nRetrieval benchmark completed successfully!")


if __name__ == "__main__":
    main()
