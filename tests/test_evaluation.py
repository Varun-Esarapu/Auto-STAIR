"""
Unit tests for Auto-STAIR Evaluation Module.
Verifies metric mathematical implementations: Recall@1, Recall@3, nDCG@3,
and Structural Hallucination Rate.
"""

import pytest
from src.evaluation.eval_searchtome import (
    calculate_recall_at_k,
    calculate_ndcg_at_k,
    calculate_hallucination_rate,
    EvaluationReport,
)


def test_recall_at_k_computation():
    """Verify Recall@1 and Recall@3 exact calculations."""
    gold_targets = [
        "1.1 Introduction",
        "2.3 Sorting Algorithms",
        "3.1 Dynamic Programming",
    ]
    predictions = [
        ["1.1 Introduction", "1.2 Scope", "1.3 Motivation"],       # Hit at rank 1
        ["2.1 Arrays", "2.3 Sorting Algorithms", "2.2 Trees"],     # Hit at rank 2
        ["4.1 Graphs", "4.2 Shortest Path", "4.3 Spanning Tree"],  # Miss
    ]

    # Recall@1: Only 1st query hits at rank 1 -> 1 / 3 = 33.33%
    r1 = calculate_recall_at_k(predictions, gold_targets, k=1)
    assert r1 == 33.33

    # Recall@3: Queries 1 and 2 hit within top 3 -> 2 / 3 = 66.67%
    r3 = calculate_recall_at_k(predictions, gold_targets, k=3)
    assert r3 == 66.67


def test_ndcg_at_k_computation():
    """Verify nDCG@3 ranking discounting."""
    gold_targets = ["Target A", "Target B"]
    
    # Query 1: hit at rank 1 -> DCG = 1/log2(2) = 1.0
    # Query 2: hit at rank 2 -> DCG = 1/log2(3) = 0.6309
    # Average nDCG = (1.0 + 0.6309) / 2 = 0.8155 -> 81.55%
    predictions = [
        ["Target A", "Other 1", "Other 2"],
        ["Other 3", "Target B", "Other 4"],
    ]

    ndcg = calculate_ndcg_at_k(predictions, gold_targets, k=3)
    assert 81.0 <= ndcg <= 82.0


def test_hallucination_rate_computation():
    """Verify structural hallucination rate detects invalid section titles."""
    valid_leaf_sets = [
        {"Section 1", "Section 2", "Section 3"},
        {"Chapter A", "Chapter B"},
    ]

    # Query 1: 3 predictions, all valid -> 0 hallucinations
    # Query 2: 2 predictions, 1 valid, 1 fabricated ("Ghost Chapter") -> 1 hallucination
    # Total predictions = 5, hallucinations = 1 -> 20.00%
    predictions = [
        ["Section 1", "Section 2", "Section 3"],
        ["Chapter A", "Ghost Chapter"],
    ]

    rate = calculate_hallucination_rate(predictions, valid_leaf_sets)
    assert rate == 20.00

    # 100% valid predictions -> 0.00% hallucination rate
    clean_preds = [
        ["Section 1", "Section 2"],
        ["Chapter A"],
    ]
    assert calculate_hallucination_rate(clean_preds, valid_leaf_sets) == 0.00


def test_evaluation_report_markdown():
    """Verify EvaluationReport formats markdown table cleanly."""
    report = EvaluationReport(
        domain="education",
        sample_size=50,
        auto_stair_r1=68.5,
        auto_stair_r3=84.2,
        auto_stair_ndcg3=79.1,
        auto_stair_hallucination=0.0,
        bm25_r1=59.5,
        bm25_r3=77.6,
        bm25_ndcg3=70.1,
        bm25_hallucination=0.0,
    )
    md = report.to_markdown_table()
    assert "Benchmark Comparison on SearchTome (Education, N=50)" in md
    assert "68.50%" in md
    assert "0.00%" in md
    assert "Zero-Shot Mistral Instruct v0.2" in md
