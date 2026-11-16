
from pathlib import Path
import argparse
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

DEFAULT_VECTOR_STORE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "apple_annual_report_embeddings.npz"
)

DEFAULT_REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "retrieval_comparison.json"
)


def parse_arguments():
    """Read benchmark configuration from command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Compare financial document retrieval systems."
    )

    parser.add_argument(
        "--dataset-file",
        type=Path,
        default=DATASET_FILE,
        help="Path to the retrieval evaluation dataset."
    )

    parser.add_argument(
        "--vector-store",
        type=Path,
        default=DEFAULT_VECTOR_STORE,
        help="Path to the document embedding vector store."
    )

    parser.add_argument(
        "--report-file",
        type=Path,
        default=DEFAULT_REPORT_FILE,
        help="Path where the benchmark report will be saved."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\nFinancial Document Retrieval Benchmark")
    print("=" * 60)

    # 1. Load evaluation dataset.
    dataset = load_evaluation_dataset(args.dataset_file)

    print(f"Evaluation questions: {len(dataset)}")
    print(f"Top K: {TOP_K}")
    print(f"Dataset: {args.dataset_file}")
    print(f"Vector store: {args.vector_store}")

    # 2. Load the selected embedding model and vector store once.
    semantic_engine = SemanticSearchEngine(
        vector_store_file=args.vector_store
    )

    print(f"Corpus chunks: {len(semantic_engine.metadata)}")

    # 3. Initialize BM25 using the exact same corpus.
    bm25_engine = BM25SearchEngine(
        metadata=semantic_engine.metadata
    )

    # 4. Initialize Hybrid using the shared semantic engine.
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
        "total_corpus_chunks": len(semantic_engine.metadata),
        "dataset_file": str(args.dataset_file),
        "vector_store_file": str(args.vector_store),
        "systems": {},
    }

    # 5. Evaluate all systems on the same dataset.
    for system_name, engine in engines.items():

        print("\n" + "=" * 60)
        print(f"EVALUATING: {system_name.upper()} SEARCH")
        print("=" * 60)

        report = evaluate_search(engine, dataset)

        comparison["systems"][system_name] = report

    # 6. Build average metric comparison.
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

    # 7. Save report without overwriting another benchmark.
    args.report_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with args.report_file.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=4)

    # 8. Display final comparison.
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
    print(args.report_file)

    print("\nRetrieval benchmark completed successfully!")


if __name__ == "__main__":
    main()
