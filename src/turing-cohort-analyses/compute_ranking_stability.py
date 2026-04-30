#!/usr/bin/env python3
"""
================================================================================
MSV: Kaggle: NeurIPS E&D 2026 -- Ranking Stability Under Item Subsampling (v2)
================================================================================
Project     : MSV Metacognition Benchmark
Paper       : "Beyond Confidence Calibration: Behavioral Metacognitive Control
               as a Distinct Evaluation Target for Large Language Models"
Track       : NeurIPS 2026 Evaluations & Datasets

Purpose
-------
Assess whether the 80-question benchmark produces stable model rankings
under item subsampling. This addresses the reviewer concern "is 80 questions
enough for reliable measurement?"

Revision Notes (v2)
-------------------
- Reframed from "split-half reliability" (which implies psychometric
  instrument reliability in the classical sense) to "ranking stability under
  item subsampling," which accurately describes what this analysis measures.
- Classical split-half reliability requires many subjects (models) per split,
  and with ~10 models the Pearson correlation across models in each split
  is computed over very few data points, making it statistically fragile.
  This limitation is now documented explicitly.
- The Spearman-Brown correction is retained but caveated: it assumes the two
  halves are parallel forms, which is not guaranteed for arbitrary random
  splits of GPQA questions.

Method
------
1. Ranking stability: randomly split the 80 questions into two halves,
   compute the target metric on each half for each model, rank models on
   each half, and compute rank-order correlation. Repeat N times and report
   the distribution of rank correlations.

2. Item discrimination: for each question, compute the point-biserial
   correlation between delegation and actual incorrectness across models.

Inputs
------
Per-model CSV files with columns:
    question_id, answer, correct, confidence, delegated, difficulty

Outputs
-------
- ranking_stability_summary.csv
- item_discrimination.csv
- ranking_stability_distribution.csv

Usage
-----
    python compute_ranking_stability.py --input_dir ./results/delegate_game/ \
                                        --output_dir ./results/stability/ \
                                        --n_splits 1000

Dependencies
------------
    numpy, pandas, scipy, scikit-learn
================================================================================
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau, pointbiserialr
from sklearn.metrics import roc_auc_score


# ============================================================================
# Configuration
# ============================================================================

BASELINE_REWARD = {
    "correct_conf4": 1.00, "correct_conf3": 1.00,
    "correct_conf2": 0.65, "correct_conf1": 0.40,
    "incorrect": 0.00,
    "delegate_hard": 0.85, "delegate_med": 0.60, "delegate_easy": 0.15,
}

BASELINE_THRESHOLDS = {"hard_above": 0.65, "easy_below": 0.40}


# ============================================================================
# Metric Computation
# ============================================================================

def _classify_difficulty(diff, thresholds=None):
    if thresholds is None:
        thresholds = BASELINE_THRESHOLDS
    if diff > thresholds["hard_above"]:
        return "hard"
    elif diff < thresholds["easy_below"]:
        return "easy"
    return "medium"


def compute_reward(df):
    """Compute mean reward score for a subset of trials."""
    scores = []
    for _, row in df.iterrows():
        if row["delegated"] == 1:
            dc = _classify_difficulty(row["difficulty"])
            key = f"delegate_{dc}" if dc != "medium" else "delegate_med"
            scores.append(BASELINE_REWARD[key])
        elif row["correct"] == 1:
            c = int(row["confidence"]) if not pd.isna(row["confidence"]) else 2
            scores.append(BASELINE_REWARD.get(f"correct_conf{c}", 0.65))
        else:
            scores.append(BASELINE_REWARD["incorrect"])
    return np.mean(scores) if scores else np.nan


def compute_delegation_auc(df):
    """AUC-ROC of delegation as predictor of incorrectness."""
    correct = df["correct"].values.astype(int)
    delegated = df["delegated"].values.astype(int)
    labels = 1 - correct
    if len(np.unique(labels)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    return roc_auc_score(labels, delegated)


# ============================================================================
# Ranking Stability Analysis
# ============================================================================

def run_ranking_stability(model_dfs, n_splits=1000, seed=42):
    """
    Assess ranking stability under random item subsampling.

    For each split, divide the common question set into two halves, compute
    the metric on each half, rank models by each half, and record the
    rank-order correlation (Spearman rho and Kendall tau) between the two
    sets of rankings.

    Note: with only ~10 models, each correlation is computed over ~10 data
    points. Individual correlations are noisy; the distribution over many
    splits is more informative than any single split.

    Returns
    -------
    rank_corrs : list of dict
        Per-split Spearman rho and Kendall tau for reward and AUC metrics.
    """
    rng = np.random.RandomState(seed)
    models = sorted(model_dfs.keys())
    n_models = len(models)

    # Common question set
    all_qids = None
    for mname, df in model_dfs.items():
        qids = set(df["question_id"].unique())
        all_qids = qids if all_qids is None else all_qids & qids

    all_qids = sorted(all_qids)
    n_questions = len(all_qids)
    half = n_questions // 2

    print(f"  Common questions: {n_questions}")
    print(f"  Split size: {half} / {n_questions - half}")
    print(f"  Models: {n_models}")
    if n_models < 5:
        print(f"  WARNING: Only {n_models} models. Rank correlations over "
              f"this few data points are statistically fragile.")

    rank_corrs = []

    for s in range(n_splits):
        perm = rng.permutation(all_qids)
        half_a = set(perm[:half])
        half_b = set(perm[half:])

        scores_a = {}
        scores_b = {}

        for mname in models:
            df = model_dfs[mname]
            df_a = df[df["question_id"].isin(half_a)]
            df_b = df[df["question_id"].isin(half_b)]
            scores_a[mname] = compute_reward(df_a)
            scores_b[mname] = compute_reward(df_b)

        # Rank models by each half
        ranked_a = sorted(models, key=lambda m: -scores_a[m])
        ranked_b = sorted(models, key=lambda m: -scores_b[m])

        rank_a = {m: i for i, m in enumerate(ranked_a)}
        rank_b = {m: i for i, m in enumerate(ranked_b)}

        ra = np.array([rank_a[m] for m in models])
        rb = np.array([rank_b[m] for m in models])

        if n_models >= 3:
            sp_rho, sp_p = spearmanr(ra, rb)
            kt_tau, kt_p = kendalltau(ra, rb)
            # Also Pearson on raw scores (not ranks)
            sa = np.array([scores_a[m] for m in models])
            sb = np.array([scores_b[m] for m in models])
            valid = ~(np.isnan(sa) | np.isnan(sb))
            if valid.sum() >= 3:
                pe_r, _ = pearsonr(sa[valid], sb[valid])
            else:
                pe_r = np.nan
        else:
            sp_rho = kt_tau = pe_r = np.nan

        rank_corrs.append({
            "split": s,
            "spearman_rho": sp_rho,
            "kendall_tau": kt_tau,
            "pearson_r_scores": pe_r,
        })

    return rank_corrs


def spearman_brown(r_half):
    """
    Spearman-Brown prophecy formula for estimated full-test reliability.

    Caveat: this assumes the two halves are parallel forms, which is not
    guaranteed for arbitrary splits of heterogeneous GPQA questions.
    """
    if np.isnan(r_half) or r_half <= -1:
        return np.nan
    return 2 * r_half / (1 + r_half)


# ============================================================================
# Item Discrimination
# ============================================================================

def compute_item_discrimination(model_dfs):
    """
    Per-question discrimination: point-biserial correlation between
    delegation (0/1) and actual incorrectness (0/1) across models.

    High positive correlation = the question discriminates well (models
    that get it wrong tend to delegate).
    """
    models = sorted(model_dfs.keys())

    all_qids = None
    for mname, df in model_dfs.items():
        qids = set(df["question_id"].unique())
        all_qids = qids if all_qids is None else all_qids & qids
    all_qids = sorted(all_qids)

    results = []
    for qid in all_qids:
        delegations = []
        incorrectness = []
        for mname in models:
            df = model_dfs[mname]
            row = df[df["question_id"] == qid]
            if len(row) == 0:
                continue
            row = row.iloc[0]
            delegations.append(int(row["delegated"]))
            incorrectness.append(1 - int(row["correct"]))

        delegations = np.array(delegations)
        incorrectness = np.array(incorrectness)

        if (len(np.unique(delegations)) < 2 or
                len(np.unique(incorrectness)) < 2):
            rpb, p_val = np.nan, np.nan
        else:
            rpb, p_val = pointbiserialr(incorrectness, delegations)

        # Difficulty from first model that has it
        diff_val = np.nan
        for mname in models:
            drow = model_dfs[mname]
            drow = drow[drow["question_id"] == qid]
            if len(drow) > 0 and "difficulty" in drow.columns:
                diff_val = drow["difficulty"].values[0]
                break

        results.append({
            "question_id": qid,
            "difficulty": diff_val,
            "delegation_rate": np.mean(delegations),
            "incorrectness_rate": np.mean(incorrectness),
            "discrimination_rpb": rpb,
            "discrimination_p": p_val,
        })

    return pd.DataFrame(results)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ranking stability under item subsampling and "
                    "item discrimination analysis."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./results/stability/")
    parser.add_argument("--n_splits", type=int, default=1000)
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

    print(f"Loaded {len(model_dfs)} model(s)")
    print(f"Running {args.n_splits} random splits\n")

    # --- Ranking Stability ---
    print("=== RANKING STABILITY UNDER ITEM SUBSAMPLING ===")
    rank_corrs = run_ranking_stability(model_dfs, n_splits=args.n_splits)

    rc_df = pd.DataFrame(rank_corrs)
    rc_df.to_csv(
        os.path.join(args.output_dir, "ranking_stability_distribution.csv"),
        index=False
    )

    mean_rho = rc_df["spearman_rho"].mean()
    mean_tau = rc_df["kendall_tau"].mean()
    mean_pr = rc_df["pearson_r_scores"].mean()
    sb = spearman_brown(mean_pr)

    print(f"\n  Mean Spearman rho:  {mean_rho:.4f}")
    print(f"  Mean Kendall tau:   {mean_tau:.4f}")
    print(f"  Mean Pearson r:     {mean_pr:.4f}")
    print(f"  Spearman-Brown*:    {sb:.4f}")
    print(f"  * Caveat: assumes parallel forms; interpret with caution.")

    summary = pd.DataFrame([{
        "metric": "reward",
        "mean_spearman_rho": mean_rho,
        "mean_kendall_tau": mean_tau,
        "mean_pearson_r": mean_pr,
        "spearman_brown_corrected": sb,
        "n_splits": args.n_splits,
        "n_models": len(model_dfs),
    }])
    summary.to_csv(
        os.path.join(args.output_dir, "ranking_stability_summary.csv"),
        index=False
    )

    # --- Item Discrimination ---
    print("\n=== ITEM DISCRIMINATION ===")
    item_df = compute_item_discrimination(model_dfs)
    item_df.to_csv(
        os.path.join(args.output_dir, "item_discrimination.csv"),
        index=False
    )

    valid = item_df.dropna(subset=["discrimination_rpb"])
    if len(valid) > 0:
        mean_rpb = valid["discrimination_rpb"].mean()
        n_good = (valid["discrimination_rpb"] > 0.3).sum()
        n_poor = (valid["discrimination_rpb"] < 0.1).sum()
        print(f"  Mean discrimination (r_pb): {mean_rpb:.4f}")
        print(f"  Good items (r_pb > 0.3):    {n_good}/{len(valid)}")
        print(f"  Poor items (r_pb < 0.1):    {n_poor}/{len(valid)}")

    print(f"\nOutputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
