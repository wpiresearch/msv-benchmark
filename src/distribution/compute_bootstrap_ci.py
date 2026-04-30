#!/usr/bin/env python3
"""
================================================================================
MSV: Kaggle: NeurIPS E&D 2026 -- Bootstrap Confidence Intervals (v2)
================================================================================
Project     : MSV Metacognition Benchmark
Paper       : "Beyond Confidence Calibration: Behavioral Metacognitive Control
               as a Distinct Evaluation Target for Large Language Models"
Track       : NeurIPS 2026 Evaluations & Datasets

Purpose
-------
Compute bootstrap 95% confidence intervals for all reported metrics. With
80 questions, point estimates alone are insufficient to support claims about
inter-model differences.

Revision Notes (v2)
-------------------
- MCC hierarchy is now explicit: MCC against own incorrectness is primary;
  MCC against dataset hardness is secondary/auxiliary.
- Each metric documents whether it is computed on all trials or
  answered-only trials.
- Added --forced_answer_dir support for declarative metrics (ECE, Brier).
- Bootstrap resampling is over QUESTIONS (not trials), since questions are
  the unit of independent sampling.

Metrics Bootstrapped
--------------------
All-trial metrics (behavioral):
    - Delegation AUC-ROC (delegation vs own incorrectness)
    - MCC (delegation vs own incorrectness, primary)
    - MCC (delegation vs dataset hardness, auxiliary)
    - Mean reward score
    - Delegation rate
    - Accuracy (computed on answered trials, but denominator is all trials
      if model delegated some)

Answered-only or forced-answer metrics (declarative):
    - ECE
    - Brier score

Inputs
------
Per-model Delegate Game CSV:
    question_id, answer, correct, confidence, delegated, difficulty

Optional forced-answer CSV:
    question_id, correct, confidence

Outputs
-------
- bootstrap_ci_summary.csv : per-model CIs
- bootstrap_pairwise.csv   : pairwise difference CIs for Delegation AUC-ROC

Usage
-----
    python compute_bootstrap_ci.py --input_dir ./results/delegate_game/ \
                                   --output_dir ./results/bootstrap/ \
                                   --n_boot 10000

Dependencies
------------
    numpy, pandas, scikit-learn
================================================================================
"""

import argparse
import glob
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, matthews_corrcoef


# ============================================================================
# Configuration
# ============================================================================

CONFIDENCE_TO_PROB = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}

BASELINE_REWARD = {
    "correct_conf4": 1.00, "correct_conf3": 1.00,
    "correct_conf2": 0.65, "correct_conf1": 0.40,
    "incorrect": 0.00,
    "delegate_hard": 0.85, "delegate_med": 0.60, "delegate_easy": 0.15,
}

BASELINE_THRESHOLDS = {"hard_above": 0.65, "easy_below": 0.40}


# ============================================================================
# Metric Functions
# ============================================================================

def _classify_difficulty(diff, thresholds=None):
    if thresholds is None:
        thresholds = BASELINE_THRESHOLDS
    if diff > thresholds["hard_above"]:
        return "hard"
    elif diff < thresholds["easy_below"]:
        return "easy"
    return "medium"


def compute_reward_score(correct, confidence, delegated, difficulty):
    """Mean per-question reward (all trials)."""
    scores = np.zeros(len(correct))
    for i in range(len(correct)):
        if delegated[i] == 1:
            dc = _classify_difficulty(difficulty[i])
            key = f"delegate_{dc}" if dc != "medium" else "delegate_med"
            scores[i] = BASELINE_REWARD[key]
        elif correct[i] == 1:
            c = int(confidence[i]) if not np.isnan(confidence[i]) else 2
            scores[i] = BASELINE_REWARD.get(f"correct_conf{c}", 0.65)
        else:
            scores[i] = BASELINE_REWARD["incorrect"]
    return np.mean(scores)


def compute_delegation_auc(correct, delegated):
    """AUC-ROC: delegation as classifier for own incorrectness (all trials)."""
    labels = 1 - correct
    if len(np.unique(labels)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    return roc_auc_score(labels, delegated)


def compute_mcc_own_error(correct, delegated):
    """MCC: delegation vs own incorrectness (PRIMARY, all trials)."""
    labels = (1 - correct).astype(int)
    if len(np.unique(labels)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    return matthews_corrcoef(labels, delegated)


def compute_mcc_hardness(delegated, difficulty):
    """MCC: delegation vs dataset hardness (AUXILIARY, all trials)."""
    hard = (difficulty > BASELINE_THRESHOLDS["hard_above"]).astype(int)
    if len(np.unique(hard)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    return matthews_corrcoef(hard, delegated)


def compute_ece(correct, confidence, n_bins=4):
    """ECE grouped by discrete confidence label.

    Per paper Section 4.3: ECE is computed with one bin per discrete
    confidence level (k in {1, 2, 3, 4}), not by equal-width binning of
    the [0, 1] interval. This avoids boundary ambiguity (which would
    silently merge confidence levels 3 and 4 under the canonical
    mapping) and matches the paper's stated estimator. The n_bins
    parameter is retained for backward compatibility but unused.

    `confidence` is expected to contain integer labels in {1, 2, 3, 4}.
    """
    del n_bins  # retained in signature for API compatibility; unused
    if len(correct) == 0:
        return np.nan
    # Filter out NaN confidence values
    valid = ~np.isnan(confidence)
    if valid.sum() == 0:
        return np.nan
    correct = correct[valid]
    confidence = confidence[valid].astype(int)
    n = len(correct)
    ece = 0.0
    for k in (1, 2, 3, 4):
        mask = (confidence == k)
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_k = CONFIDENCE_TO_PROB[k]
        acc_k = correct[mask].mean()
        ece += (nb / n) * abs(acc_k - p_k)
    return ece


def compute_brier(correct, confidence):
    """Brier score on the provided trials."""
    if len(correct) == 0:
        return np.nan
    valid = ~np.isnan(confidence)
    if valid.sum() == 0:
        return np.nan
    correct = correct[valid]
    confidence = confidence[valid]
    conf_prob = np.array([CONFIDENCE_TO_PROB.get(int(c), 0.5) for c in confidence])
    return np.mean((conf_prob - correct.astype(float)) ** 2)


# ============================================================================
# Bootstrap Engine
# ============================================================================

def bootstrap_metric(metric_fn, data_arrays, n_boot=10000, ci=0.95,
                     seed=42):
    """
    Bootstrap CI for a metric function. Resamples over questions (rows).

    Returns dict with: point, ci_lo, ci_hi, se
    """
    rng = np.random.RandomState(seed)
    n = len(data_arrays[0])
    point = metric_fn(*data_arrays)

    boot_values = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        resampled = [arr[idx] for arr in data_arrays]
        try:
            boot_values[b] = metric_fn(*resampled)
        except (ValueError, ZeroDivisionError):
            boot_values[b] = np.nan

    boot_values = boot_values[~np.isnan(boot_values)]
    if len(boot_values) == 0:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "se": np.nan}

    alpha = 1 - ci
    lo = np.percentile(boot_values, 100 * alpha / 2)
    hi = np.percentile(boot_values, 100 * (1 - alpha / 2))

    return {"point": point, "ci_lo": lo, "ci_hi": hi, "se": np.std(boot_values)}


# ============================================================================
# Per-Model Analysis
# ============================================================================

def analyze_model_bootstrap(df, model_name, n_boot=10000, fa_df=None):
    """Compute bootstrap CIs for all metrics for one model."""
    correct = df["correct"].values.astype(int)
    confidence = df["confidence"].fillna(2).values.astype(float)
    delegated = df["delegated"].values.astype(int)
    difficulty = df["difficulty"].values.astype(float)

    results = {"model": model_name}

    # --- All-trial behavioral metrics ---
    for metric_name, fn, arrays in [
        ("reward", compute_reward_score,
         [correct, confidence, delegated, difficulty]),
        ("deleg_auc", compute_delegation_auc, [correct, delegated]),
        ("mcc_own_error", compute_mcc_own_error, [correct, delegated]),
        ("mcc_hardness", compute_mcc_hardness, [delegated, difficulty]),
        ("deleg_rate", lambda d: np.mean(d), [delegated]),
    ]:
        res = bootstrap_metric(fn, arrays, n_boot=n_boot)
        results[f"{metric_name}_point"] = res["point"]
        results[f"{metric_name}_ci_lo"] = res["ci_lo"]
        results[f"{metric_name}_ci_hi"] = res["ci_hi"]
        results[f"{metric_name}_se"] = res["se"]

    # --- Declarative metrics ---
    if fa_df is not None:
        # Use forced-answer data for all items
        merged = df[["question_id"]].merge(
            fa_df[["question_id", "correct", "confidence"]],
            on="question_id", how="inner"
        )
        decl_correct = merged["correct"].values.astype(int)
        decl_conf = merged["confidence"].values.astype(float)
        results["declarative_source"] = "forced_answer"
    else:
        # Answered-only
        answered = df[delegated == 0]
        decl_correct = answered["correct"].values.astype(int)
        decl_conf = answered["confidence"].values.astype(float)
        results["declarative_source"] = "answered_only"

    # Accuracy
    res = bootstrap_metric(lambda c: np.mean(c), [decl_correct], n_boot)
    results["accuracy_point"] = res["point"]
    results["accuracy_ci_lo"] = res["ci_lo"]
    results["accuracy_ci_hi"] = res["ci_hi"]

    if len(decl_correct) > 0:
        for metric_name, fn in [("ece", compute_ece), ("brier", compute_brier)]:
            res = bootstrap_metric(fn, [decl_correct, decl_conf], n_boot)
            results[f"{metric_name}_point"] = res["point"]
            results[f"{metric_name}_ci_lo"] = res["ci_lo"]
            results[f"{metric_name}_ci_hi"] = res["ci_hi"]
    else:
        for metric_name in ["ece", "brier"]:
            results[f"{metric_name}_point"] = np.nan
            results[f"{metric_name}_ci_lo"] = np.nan
            results[f"{metric_name}_ci_hi"] = np.nan

    return results


# ============================================================================
# Pairwise Comparison
# ============================================================================

def pairwise_bootstrap(model_dfs, n_boot=10000):
    """Bootstrap CIs on Delegation AUC-ROC differences between model pairs."""
    models = sorted(model_dfs.keys())
    pairwise = []

    for m1, m2 in combinations(models, 2):
        df1, df2 = model_dfs[m1], model_dfs[m2]
        common = set(df1["question_id"]) & set(df2["question_id"])
        if len(common) < 10:
            continue

        d1 = df1[df1["question_id"].isin(common)].sort_values("question_id")
        d2 = df2[df2["question_id"].isin(common)].sort_values("question_id")

        def _auc(df):
            c = df["correct"].values.astype(int)
            d = df["delegated"].values.astype(int)
            return compute_delegation_auc(c, d)

        point_diff = _auc(d1) - _auc(d2)

        rng = np.random.RandomState(42)
        n = len(d1)
        diffs = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.randint(0, n, size=n)
            try:
                diffs[b] = _auc(d1.iloc[idx]) - _auc(d2.iloc[idx])
            except (ValueError, ZeroDivisionError):
                diffs[b] = np.nan

        diffs = diffs[~np.isnan(diffs)]
        if len(diffs) == 0:
            continue

        lo, hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

        pairwise.append({
            "model_a": m1, "model_b": m2,
            "diff_point": point_diff,
            "diff_ci_lo": lo, "diff_ci_hi": hi,
            "significant_at_05": (lo > 0) or (hi < 0),
        })

    return pairwise


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap CIs for all Delegate Game and comparative metrics."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--forced_answer_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./results/bootstrap/")
    parser.add_argument("--n_boot", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSVs in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    model_dfs = {}
    for fpath in csv_files:
        mname = os.path.splitext(os.path.basename(fpath))[0]
        try:
            model_dfs[mname] = pd.read_csv(fpath)
        except Exception as e:
            print(f"WARNING: {fpath}: {e}", file=sys.stderr)

    fa_data = {}
    if args.forced_answer_dir:
        for fpath in glob.glob(os.path.join(args.forced_answer_dir, "*.csv")):
            mname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                fa_data[mname] = pd.read_csv(fpath)
            except Exception as e:
                print(f"WARNING (FA): {fpath}: {e}", file=sys.stderr)

    print(f"Models: {len(model_dfs)}, Forced-answer: {len(fa_data)}")
    print(f"Bootstrap iterations: {args.n_boot}\n")

    all_results = []
    for mname, df in sorted(model_dfs.items()):
        print(f"  Bootstrapping: {mname}")
        fa_df = fa_data.get(mname, None)
        res = analyze_model_bootstrap(df, mname, args.n_boot, fa_df)
        all_results.append(res)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(
        os.path.join(args.output_dir, "bootstrap_ci_summary.csv"), index=False
    )
    print(f"\nPer-model CIs saved.")

    print("\nPairwise bootstrap for Delegation AUC-ROC...")
    pw = pairwise_bootstrap(model_dfs, args.n_boot)
    if pw:
        pw_df = pd.DataFrame(pw)
        pw_df.to_csv(
            os.path.join(args.output_dir, "bootstrap_pairwise.csv"), index=False
        )
        n_sig = pw_df["significant_at_05"].sum()
        print(f"  {n_sig}/{len(pw_df)} pairs significant at alpha=0.05")

    print("\nDone.")


if __name__ == "__main__":
    main()
