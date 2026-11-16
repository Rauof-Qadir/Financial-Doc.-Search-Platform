
import pytest

from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    hit_rate_at_k
)


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "F"]

    assert precision_at_k(retrieved, relevant, 5) == 0.4


def test_recall_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    relevant = ["A", "C", "F"]

    assert recall_at_k(retrieved, relevant, 5) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["C"]

    assert reciprocal_rank(retrieved, relevant) == pytest.approx(1 / 3)


def test_reciprocal_rank_when_no_relevant_result():
    retrieved = ["A", "B", "C"]
    relevant = ["Z"]

    assert reciprocal_rank(retrieved, relevant) == 0.0


def test_hit_rate_at_k():
    retrieved = ["A", "B", "C", "D"]
    relevant = ["C"]

    assert hit_rate_at_k(retrieved, relevant, 3) == 1.0


def test_hit_rate_when_no_relevant_result():
    retrieved = ["A", "B", "C"]
    relevant = ["Z"]

    assert hit_rate_at_k(retrieved, relevant, 3) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["A"], ["A"], 0)


def test_empty_relevant_ids():
    with pytest.raises(ValueError):
        recall_at_k(["A"], [], 5)
