#!/usr/bin/env python3
"""
================================================================================
MSV: Kaggle: NeurIPS E&D 2026 -- Comparative Evaluation Baselines (v2)
================================================================================
Project     : MSV Metacognition Benchmark
Paper       : "Beyond Confidence Calibration: Behavioral Metacognitive Control
               as a Distinct Evaluation Target for Large Language Models"
Track       : NeurIPS 2026 Evaluations & Datasets

Purpose
-------
Compute declarative confidence baselines from Delegate Game outputs so that
behavioral and declarative evaluation metrics can be compared on the same
models and questions. This is the central comparative analysis for the
NeurIPS E&D submission (Section 5.2).

Revision Notes (v2)
-------------------
- Added --forced_answer_dir for a two-phase design: Phase 1 (forced-answer,
  no delegation) provides the declarative confidence signal for ALL items,
  including those the model would have delegated in Phase 2. This eliminates
  the circularity of imputing confidence=0 for delegated items.
- When forced-answer data is unavailable, ECE/Brier/Abstention AUC are
  computed on ANSWERED TRIALS ONLY with an explicit warning. Delegated items
  are excluded entirely rather than imputed.
- ECE now defaults to 4 bins matching the 4 discrete confidence levels,
  since 5 equal-width bins on a 4-level scale leaves degenerate bins.
- MCC hierarchy is explicit: Delegation AUC-ROC against own incorrectness
  is primary; MCC against dataset-level hardness is secondary/auxiliary.
- Confidence-to-probability mapping is documented as a design assumption
  with sensitivity noted.

Metrics Computed
----------------
1. Expected Calibration Error (ECE) -- binned by confidence level (4 bins
   matching the 4 confidence levels, unless overridden).
2. Brier Score -- mean squared difference between stated confidence
   (normalized) and binary correctness.
3. Selective Prediction Accuracy at multiple confidence thresholds.
4. Abstention AUC -- confidence as a classifier for incorrectness.
   When forced-answer data is available, this uses forced-answer confidence
   for ALL items. When unavailable, uses answered-only data.

Inputs
------
Delegate Game CSV (one per model) with columns:
    question_id, answer, correct, confidence, delegated, difficulty

Optional forced-answer CSV (one per model, same naming convention):
    question_id, answer, correct, confidence
    (No delegation column; the model was forced to answer every question.)

Outputs
-------
- comparative_summary.csv : one row per model with all metrics
- per-model selective prediction detail CSVs

Usage
-----
    # With forced-answer data (preferred, methodologically clean):
    python compute_comparative_baselines.py \
        --input_dir ./results/delegate_game/ \
        --forced_answer_dir ./results/forced_answer/ \
        --output_dir ./results/comparative/

    # Without forced-answer data (answered-only, with warning):
    python compute_comparative_baselines.py \
        --input_dir ./results/delegate_game/ \
        --output_dir ./results/comparative/

Dependencies
------------
    numpy, pandas, scikit-learn
================================================================================
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


# ============================================================================
# Configuration
# ============================================================================

# Default confidence-to-probability mapping (linear, equally spaced).
# This is a design choice documented in the paper. Alternative mappings
# (logistic) can be tested via the --confidence_map flag.
CONFIDENCE_TO_PROB = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

# ECE defaults to 4 bins matching the 4 discrete confidence levels.
# With only 4 possible values, equal-width binning with 5+ bins creates
# degenerate (always-empty) bins, which is methodologically unsound.
ECE_N_BINS = 4

SELECTIVE_THRESHOLDS = [1, 2, 3, 4]


# ============================================================================
# Metric Computation
# ============================================================================

def compute_ece(correctness: np.ndarray, confidence_probs: np.ndarray,
                n_bins: int = ECE_N_BINS) -> float:
    """
    Compute Expected Calibration Error grouped by discrete confidence label.

    Per paper Section 4.3: ECE is computed with one bin per discrete
    confidence level (k in {1, 2, 3, 4}), not by equal-width binning of
    the [0, 1] interval. The previous implementation used four
    equal-width bins on [0, 1], which under the canonical mapping
    {0.25, 0.50, 0.75, 1.00} silently merged confidence levels 3 and 4
    into the last bin (and left the [0.00, 0.25) bin empty). The
    label-grouped form below avoids boundary ambiguity and matches the
    paper's stated estimator. The n_bins parameter is retained for
    backward compatibility but unused.

    Inputs: confidence_probs are float values that should be in
    {0.25, 0.50, 0.75, 1.00} after the canonical mapping has been
    applied. Discrete labels are recovered via round(prob * 4).
    """
    del n_bins  # retained in signature for API compatibility; unused
    n_total = len(correctness)
    if n_total == 0:
        return np.nan
    # Recover integer confidence labels: prob 0.25 -> 1, 0.50 -> 2, 0.75 -> 3, 1.00 -> 4
    labels = np.rint(np.asarray(confidence_probs, dtype=float) * 4).astype(int)
    correctness = np.asarray(correctness, dtype=float)
    ece = 0.0
    for k in (1, 2, 3, 4):
        mask = (labels == k)
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        p_k = k / 4.0
        avg_accuracy = correctness[mask].mean()
        ece += (n_in_bin / n_total) * abs(avg_accuracy - p_k)
    return ece


def compute_brier(correctness: np.ndarray,
                  confidence_probs: np.ndarray) -> float:
    """Brier Score: MSE between stated confidence and binary correctness."""
    if len(correctness) == 0:
        return np.nan
    return np.mean((confidence_probs - correctness.astype(float)) ** 2)


def compute_selective_prediction(correctness: np.ndarray,
                                 confidence_raw: np.ndarray,
                                 thresholds: list = None) -> list:
    """Selective prediction accuracy and coverage at each threshold."""
    if thresholds is None:
        thresholds = SELECTIVE_THRESHOLDS

    results = []
    n_total = len(correctness)

    for t in thresholds:
        mask = confidence_raw >= t
        n_answered = mask.sum()
        acc = correctness[mask].mean() if n_answered > 0 else np.nan

        results.append({
            "threshold": t,
            "accuracy": acc,
            "coverage": n_answered / n_total if n_total > 0 else 0.0,
            "n_answered": int(n_answered)
        })

    return results


def compute_abstention_auc(correctness: np.ndarray,
                           confidence_raw: np.ndarray) -> float:
    """
    Abstention AUC: treat low confidence as a predictor of incorrectness.

    This function operates on arrays where EVERY item has a valid confidence
    value. When using forced-answer data, all items have confidence from the
    forced-answer run. When forced-answer data is unavailable, only
    answered items are included.

    The score is NEGATIVE confidence (low confidence = more likely incorrect).
    The label is 1 = incorrect.
    """
    labels = (1 - correctness).astype(int)  # 1 = incorrect

    if len(np.unique(labels)) < 2:
        return np.nan

    scores = -confidence_raw.astype(float)
    return roc_auc_score(labels, scores)


# ============================================================================
# File I/O
# ============================================================================

def load_delegate_game(filepath: str) -> pd.DataFrame:
    """Load a Delegate Game CSV."""
    df = pd.read_csv(filepath)
    required = ["question_id", "answer", "correct", "confidence",
                "delegated", "difficulty"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")
    return df


def load_forced_answer(filepath: str) -> pd.DataFrame:
    """Load a forced-answer CSV (no delegation column)."""
    df = pd.read_csv(filepath)
    required = ["question_id", "correct", "confidence"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")
    return df


def extract_model_name(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0]


# ============================================================================
# Analysis Pipeline
# ============================================================================

def analyze_model(dg_df: pd.DataFrame, model_name: str,
                  fa_df: pd.DataFrame = None) -> tuple:
    """
    Run all comparative baseline metrics for a single model.

    Parameters
    ----------
    dg_df : pd.DataFrame
        Delegate Game results.
    model_name : str
        Model identifier.
    fa_df : pd.DataFrame or None
        Forced-answer results (Phase 1). If provided, declarative metrics
        are computed from these outputs for ALL items. If None, declarative
        metrics are computed from answered-only Delegate Game trials.
    """
    n_total = len(dg_df)
    n_delegated = (dg_df["delegated"] == 1).sum()
    n_answered = n_total - n_delegated

    # --- Delegation rate ---
    delegation_rate = n_delegated / n_total if n_total > 0 else 0.0

    # --- Delegation AUC-ROC (primary behavioral metric) ---
    # This uses ALL trials: delegation decision vs own incorrectness
    all_correct = dg_df["correct"].values.astype(int)
    all_delegated = dg_df["delegated"].values.astype(int)
    labels = (1 - all_correct).astype(int)

    if len(np.unique(labels)) >= 2 and len(np.unique(all_delegated)) >= 2:
        deleg_auc = roc_auc_score(labels, all_delegated)
    else:
        deleg_auc = np.nan

    # --- MCC against own incorrectness (auxiliary behavioral, secondary) ---
    # Note: this is Matthews Correlation Coefficient between delegation
    # and the model's own actual incorrectness, NOT against dataset-level
    # difficulty. Dataset-level difficulty MCC is computed separately below.
    from sklearn.metrics import matthews_corrcoef
    if len(np.unique(labels)) >= 2 and len(np.unique(all_delegated)) >= 2:
        mcc_own_error = matthews_corrcoef(labels, all_delegated)
    else:
        mcc_own_error = np.nan

    # --- MCC against dataset hardness (auxiliary, secondary) ---
    hard_mask = (dg_df["difficulty"] > 0.65).values.astype(int)
    if len(np.unique(hard_mask)) >= 2 and len(np.unique(all_delegated)) >= 2:
        mcc_hardness = matthews_corrcoef(hard_mask, all_delegated)
    else:
        mcc_hardness = np.nan

    # --- Declarative metrics ---
    if fa_df is not None:
        # Phase 1 data available: use forced-answer confidence for ALL items
        # Merge on question_id to align
        merged = dg_df[["question_id", "difficulty"]].merge(
            fa_df[["question_id", "correct", "confidence"]],
            on="question_id", how="inner", suffixes=("_dg", "_fa")
        )
        decl_correct = merged["correct"].values.astype(int)
        decl_conf_raw = merged["confidence"].values.astype(int)
        decl_source = "forced_answer"
        accuracy = decl_correct.mean()
    else:
        # No Phase 1 data: use answered trials only (exclude delegated)
        answered = dg_df[dg_df["delegated"] == 0].copy()
        if len(answered) == 0:
            return _empty_result(model_name, n_total, n_delegated,
                                 delegation_rate, deleg_auc, mcc_own_error,
                                 mcc_hardness), []
        decl_correct = answered["correct"].values.astype(int)
        decl_conf_raw = answered["confidence"].values.astype(int)
        decl_source = "answered_only"
        accuracy = decl_correct.mean()

    decl_conf_prob = np.array(
        [CONFIDENCE_TO_PROB.get(c, 0.5) for c in decl_conf_raw]
    )

    ece = compute_ece(decl_correct, decl_conf_prob)
    brier = compute_brier(decl_correct, decl_conf_prob)
    selective = compute_selective_prediction(decl_correct, decl_conf_raw)
    abstention_auc = compute_abstention_auc(decl_correct, decl_conf_raw)

    result = {
        "model": model_name,
        "n_total": n_total,
        "n_answered": n_answered,
        "n_delegated": n_delegated,
        "accuracy": accuracy,
        "delegation_rate": delegation_rate,
        "declarative_source": decl_source,
        "ece": ece,
        "brier": brier,
        "abstention_auc": abstention_auc,
        "delegation_auc": deleg_auc,
        "mcc_own_error": mcc_own_error,
        "mcc_hardness": mcc_hardness,
    }

    for entry in selective:
        t = entry["threshold"]
        result[f"sel_acc_t{t}"] = entry["accuracy"]
        result[f"sel_cov_t{t}"] = entry["coverage"]

    return result, selective


def _empty_result(model_name, n_total, n_delegated, delegation_rate,
                  deleg_auc, mcc_own_error, mcc_hardness):
    """Return a result dict when no answered trials are available."""
    return {
        "model": model_name, "n_total": n_total, "n_answered": 0,
        "n_delegated": n_delegated, "accuracy": np.nan,
        "delegation_rate": delegation_rate,
        "declarative_source": "none",
        "ece": np.nan, "brier": np.nan, "abstention_auc": np.nan,
        "delegation_auc": deleg_auc,
        "mcc_own_error": mcc_own_error, "mcc_hardness": mcc_hardness,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute comparative evaluation baselines (ECE, Brier, "
                    "selective prediction, abstention AUC) from Delegate Game "
                    "outputs, optionally using forced-answer Phase 1 data."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with per-model Delegate Game CSVs.")
    parser.add_argument("--forced_answer_dir", type=str, default=None,
                        help="Directory with per-model forced-answer CSVs "
                             "(same filename convention). If provided, "
                             "declarative metrics use these outputs for all "
                             "items, eliminating delegated-item circularity.")
    parser.add_argument("--output_dir", type=str,
                        default="./results/comparative/")
    parser.add_argument("--confidence_map", type=str, default="linear",
                        choices=["linear", "logistic"],
                        help="Confidence-to-probability mapping.")
    parser.add_argument("--ece_bins", type=int, default=4,
                        help="Number of ECE bins (default: 4, matching the "
                             "4 discrete confidence levels).")
    args = parser.parse_args()

    global ECE_N_BINS
    ECE_N_BINS = args.ece_bins

    if args.confidence_map == "logistic":
        def _logistic(c, midpoint=2.5, scale=1.5):
            return 1.0 / (1.0 + np.exp(-(c - midpoint) / scale))
        for c in [1, 2, 3, 4]:
            CONFIDENCE_TO_PROB[c] = _logistic(c)

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Load forced-answer data if available
    fa_data = {}
    if args.forced_answer_dir:
        fa_files = glob.glob(os.path.join(args.forced_answer_dir, "*.csv"))
        for fpath in fa_files:
            mname = extract_model_name(fpath)
            try:
                fa_data[mname] = load_forced_answer(fpath)
            except ValueError as e:
                print(f"  WARNING (forced-answer): {e}", file=sys.stderr)

    if args.forced_answer_dir and not fa_data:
        print("WARNING: --forced_answer_dir specified but no valid CSVs "
              "found. Falling back to answered-only mode.", file=sys.stderr)

    if not fa_data:
        print("NOTE: No forced-answer data provided. Declarative metrics "
              "(ECE, Brier, Abstention AUC) will be computed on ANSWERED "
              "trials only. Delegated items are excluded. For a "
              "methodologically clean comparison, provide forced-answer "
              "Phase 1 data via --forced_answer_dir.\n")

    print(f"Found {len(csv_files)} Delegate Game file(s)")
    print(f"Forced-answer data: {len(fa_data)} model(s)")
    print(f"Confidence mapping: {args.confidence_map}")
    print(f"ECE bins: {ECE_N_BINS}\n")

    summaries = []

    for fpath in csv_files:
        model_name = extract_model_name(fpath)
        print(f"Processing: {model_name}")

        try:
            dg_df = load_delegate_game(fpath)
        except ValueError as e:
            print(f"  SKIPPED: {e}", file=sys.stderr)
            continue

        fa_df = fa_data.get(model_name, None)
        if fa_df is not None:
            print(f"  Using forced-answer data ({len(fa_df)} items)")
        else:
            print(f"  No forced-answer data; using answered-only "
                  f"({(dg_df['delegated'] == 0).sum()} items)")

        result, selective = analyze_model(dg_df, model_name, fa_df)
        summaries.append(result)

        if selective:
            sel_df = pd.DataFrame(selective)
            sel_path = os.path.join(args.output_dir,
                                   f"{model_name}_selective_prediction.csv")
            sel_df.to_csv(sel_path, index=False)

        print(f"  Accuracy:           {result['accuracy']:.4f}")
        print(f"  Delegation Rate:    {result['delegation_rate']:.4f}")
        print(f"  ECE:                {result['ece']:.4f}")
        print(f"  Brier:              {result['brier']:.4f}")
        print(f"  Abstention AUC:     {result['abstention_auc']:.4f}")
        print(f"  Delegation AUC:     {result['delegation_auc']:.4f}")
        print(f"  MCC (own error):    {result['mcc_own_error']:.4f}")
        print(f"  MCC (hardness):     {result['mcc_hardness']:.4f}")
        print(f"  Source:             {result['declarative_source']}")
        print()

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_path = os.path.join(args.output_dir, "comparative_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
    else:
        print("No models processed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
