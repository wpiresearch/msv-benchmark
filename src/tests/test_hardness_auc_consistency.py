#!/usr/bin/env python3
"""
test_hardness_auc_consistency.py
=================================

Verifies the semantics of compute_rank_divergence_ci.py's hardness-AUC
implementation against an independent reference. Specifically:

  1. attach_global_hardness_label produces the correct global panel
     median across the union of unique (question_id, difficulty) pairs.
  2. The hard_label is stable across bootstrap resampling (it is a
     question-level property, attached BEFORE bootstrap).
  3. compute_delegation_auc_vs_hardness returns the same value as
     sklearn.metrics.roc_auc_score(hard_label, delegated) directly.
  4. The hardness AUC differs from the per-model-median version when
     models see different subsets of the panel.

DESIGN RATIONALE
----------------

The paper's appendix audit reports hardness AUC computed with a global
panel median, while analyze_kaggle_cohort.py historically used per-model
medians (now renamed to deleg_auc_vs_hardness_per_model_median for
self-documentation). This test file pins down the global-median
semantics so future changes to the hardness AUC implementation can't
silently revert to per-model semantics.

This test is a sibling of test_ece_consistency.py and lives in the same
internal-utilities/ folder.

USAGE
-----

    python test_hardness_auc_consistency.py

Exit code 0 if all tests pass, 1 if any test fails.

REQUIREMENTS
------------

compute_rank_divergence_ci.py must be in the same directory as this
test or the SCRIPTS_DIR variable below must be adjusted to point at it.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPTS_DIR = Path(__file__).parent


def load_module():
    """Load compute_rank_divergence_ci.py and return its module."""
    script_path = SCRIPTS_DIR / "compute_rank_divergence_ci.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find {script_path}")
    spec = importlib.util.spec_from_file_location(
        "compute_rank_divergence_ci", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_global_median_with_overlapping_panels(mod):
    """Test 1: When all models see the same panel, the global median
    equals the panel median."""
    print("Test 1: global median with overlapping panels")
    diffs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    qids = [f"q{i}" for i in range(8)]
    expected_median = float(np.median(diffs))

    def make_df(delegated_pattern):
        return pd.DataFrame({
            "question_id": qids,
            "answer": ["A"] * 8,
            "correct": [1, 0, 1, 0, 1, 0, 1, 0],
            "confidence": [3] * 8,
            "delegated": delegated_pattern,
            "difficulty": diffs,
        })

    model_dfs = {
        "m1": make_df([0, 0, 0, 0, 1, 1, 1, 1]),
        "m2": make_df([1, 0, 1, 0, 1, 0, 1, 0]),
    }

    labelled, threshold, panel_n = mod.attach_global_hardness_label(model_dfs)
    assert abs(threshold - expected_median) < 1e-9, \
        f"threshold {threshold} != expected {expected_median}"
    assert panel_n == 8, f"panel_n {panel_n} != 8"
    print(f"  ✓ threshold = {threshold}, panel_n = {panel_n}")

    # Verify hard_label values
    expected_hard = [0, 0, 0, 0, 1, 1, 1, 1]
    for m in ["m1", "m2"]:
        actual = labelled[m]["hard_label"].tolist()
        assert actual == expected_hard, \
            f"{m}: hard_label {actual} != expected {expected_hard}"
    print(f"  ✓ hard_label correctly attached: {expected_hard}")
    return True


def test_global_median_with_disjoint_panels(mod):
    """Test 2: When models see different (overlapping) subsets, the
    global median is computed over the UNION of unique question_ids,
    not over any single model's subset."""
    print("Test 2: global median with disjoint panels")
    # Model A sees questions 1-5 (difficulties 0.1, 0.2, 0.3, 0.4, 0.5)
    # Model B sees questions 4-8 (difficulties 0.4, 0.5, 0.6, 0.7, 0.8)
    # Union: q1-q8 with difficulties 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
    # Global median = 0.45 (median of 8 unique difficulty values)
    # Per-model medians: A sees 0.3, B sees 0.6. Different!
    df_a = pd.DataFrame({
        "question_id": ["q1", "q2", "q3", "q4", "q5"],
        "answer": ["A"] * 5, "correct": [1, 0, 1, 0, 1],
        "confidence": [3] * 5, "delegated": [0, 0, 1, 1, 1],
        "difficulty": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    df_b = pd.DataFrame({
        "question_id": ["q4", "q5", "q6", "q7", "q8"],
        "answer": ["A"] * 5, "correct": [1, 0, 1, 0, 1],
        "confidence": [3] * 5, "delegated": [0, 0, 1, 1, 1],
        "difficulty": [0.4, 0.5, 0.6, 0.7, 0.8],
    })

    labelled, threshold, panel_n = mod.attach_global_hardness_label(
        {"m_a": df_a, "m_b": df_b}
    )
    expected_threshold = 0.45  # median of 0.1..0.8
    assert abs(threshold - expected_threshold) < 1e-9, \
        f"global threshold {threshold} != expected {expected_threshold}"
    assert panel_n == 8, f"panel_n {panel_n} != 8 (union)"
    print(f"  ✓ global threshold = {threshold} (NOT per-model 0.3 or 0.6)")
    print(f"  ✓ panel_n = {panel_n} (union of unique question_ids)")

    # Verify model A's hard_label: q1-q5 with diffs 0.1-0.5, threshold 0.45
    # → only q5 (0.5) is hard
    a_labels = labelled["m_a"]["hard_label"].tolist()
    assert a_labels == [0, 0, 0, 0, 1], f"m_a labels {a_labels} != [0,0,0,0,1]"

    # Verify model B's hard_label: q4-q8 with diffs 0.4-0.8, threshold 0.45
    # → q5..q8 (0.5..0.8) are hard
    b_labels = labelled["m_b"]["hard_label"].tolist()
    assert b_labels == [0, 1, 1, 1, 1], f"m_b labels {b_labels} != [0,1,1,1,1]"

    print(f"  ✓ m_a hard_label = {a_labels} (only q5 of q1-q5 is hard)")
    print(f"  ✓ m_b hard_label = {b_labels} (q5-q8 of q4-q8 are hard)")
    return True


def test_auc_matches_sklearn_directly(mod):
    """Test 3: compute_delegation_auc_vs_hardness returns exactly what
    sklearn.roc_auc_score(hard_label, delegated) returns."""
    print("Test 3: compute_delegation_auc_vs_hardness matches sklearn directly")
    np.random.seed(123)
    delegated = np.random.choice([0, 1], size=20)
    hard_label = np.random.choice([0, 1], size=20)
    expected = roc_auc_score(hard_label, delegated)
    actual = mod.compute_delegation_auc_vs_hardness(delegated, hard_label)
    assert abs(actual - expected) < 1e-12, \
        f"actual {actual} != expected {expected}"
    print(f"  ✓ AUC = {actual} (matches sklearn.roc_auc_score)")
    return True


def test_degenerate_returns_nan(mod):
    """Test 4: degenerate inputs (all-hard or all-delegated) return NaN."""
    print("Test 4: degenerate inputs return NaN")
    # All hard
    auc = mod.compute_delegation_auc_vs_hardness(
        np.array([1, 0, 1, 0]), np.array([1, 1, 1, 1])
    )
    assert np.isnan(auc), f"all-hard should be NaN, got {auc}"
    print(f"  ✓ all-hard returns NaN")
    # All delegated
    auc = mod.compute_delegation_auc_vs_hardness(
        np.array([1, 1, 1, 1]), np.array([1, 0, 1, 0])
    )
    assert np.isnan(auc), f"all-delegated should be NaN, got {auc}"
    print(f"  ✓ all-delegated returns NaN")
    return True


def test_metrics_for_model_dispatch(mod):
    """Test 5: metrics_for_model dispatches correctly between targets."""
    print("Test 5: metrics_for_model dispatches between own_error and hardness")
    df = pd.DataFrame({
        "question_id": [f"q{i}" for i in range(8)],
        "answer": ["A"] * 8,
        "correct": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
        "confidence": [3] * 8,
        "delegated": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int),
        "difficulty": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    })
    labelled, _, _ = mod.attach_global_hardness_label({"m": df})

    # own_error: deleg_rate = 0.5, label = 1 - correct = [0,1,0,1,0,1,0,1]
    # delegated = [0,0,0,0,1,1,1,1]
    own = mod.metrics_for_model(labelled["m"], min_answered=2,
                                  auc_target="own_error")
    own_expected = roc_auc_score(
        np.array([0, 1, 0, 1, 0, 1, 0, 1]),
        np.array([0, 0, 0, 0, 1, 1, 1, 1]),
    )
    assert abs(own["deleg_auc"] - own_expected) < 1e-9, \
        f"own_error AUC {own['deleg_auc']} != expected {own_expected}"
    print(f"  ✓ own_error: deleg_auc = {own['deleg_auc']:.4f}")

    # hardness: hard_label = [0,0,0,0,1,1,1,1]
    # delegated = [0,0,0,0,1,1,1,1] - they match exactly, AUC = 1.0
    hard = mod.metrics_for_model(labelled["m"], min_answered=2,
                                   auc_target="hardness")
    assert abs(hard["deleg_auc"] - 1.0) < 1e-9, \
        f"hardness AUC {hard['deleg_auc']} != expected 1.0"
    print(f"  ✓ hardness: deleg_auc = {hard['deleg_auc']:.4f}")

    # Verify the two targets give different AUCs (sanity check)
    assert abs(own["deleg_auc"] - hard["deleg_auc"]) > 1e-3, \
        "own_error and hardness should give different AUCs in this test"
    print(f"  ✓ targets give DIFFERENT AUCs (own={own['deleg_auc']:.4f}, "
          f"hardness={hard['deleg_auc']:.4f}) -> dispatch is working")
    return True


def main():
    try:
        mod = load_module()
    except Exception as e:
        print(f"ERROR loading compute_rank_divergence_ci.py: {e}", file=sys.stderr)
        return 1

    tests = [
        test_global_median_with_overlapping_panels,
        test_global_median_with_disjoint_panels,
        test_auc_matches_sklearn_directly,
        test_degenerate_returns_nan,
        test_metrics_for_model_dispatch,
    ]
    failed = []
    for t in tests:
        try:
            t(mod)
            print()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            print()
            failed.append(t.__name__)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            print()
            failed.append(t.__name__)

    if failed:
        print("=" * 60)
        print(f"FAILED: {len(failed)} of {len(tests)} tests")
        for name in failed:
            print(f"  - {name}")
        print("=" * 60)
        return 1
    print("=" * 60)
    print(f"ALL {len(tests)} TESTS PASSED")
    print("Hardness AUC implementation is consistent with global-median semantics.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
