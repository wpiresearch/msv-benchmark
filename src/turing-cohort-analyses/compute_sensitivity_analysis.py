#!/usr/bin/env python3
"""
================================================================================
MSV: Kaggle: NeurIPS E&D 2026 -- Reward-Sensitivity Analysis
================================================================================
Project     : MSV Metacognition Benchmark
Paper       : "Beyond Confidence Calibration: Behavioral Metacognitive Control
               as a Distinct Evaluation Target for Large Language Models"
Track       : NeurIPS 2026 Evaluations & Datasets

Purpose
-------
Test whether the Delegate Game's model rankings are robust to perturbations
of the reward schedule and difficulty thresholds. This addresses a key
reviewer concern: are the conclusions artifacts of one particular set of
hand-tuned scoring parameters?

For the NeurIPS paper (Section 5.4), we need to show that the behavioral-vs-
declarative evaluation divergence holds across multiple reasonable reward
configurations.

Analysis
--------
1. Reward-weight perturbation: Re-score all models under N alternative
   reward schedules that vary delegation rewards, confidence bonuses,
   and incorrect-answer penalties within reasonable ranges.
2. Difficulty-threshold perturbation: Re-classify questions as easy/medium/
   hard under alternative cutoffs and re-score accordingly.
3. Rank-order stability: Compute Kendall's tau and Spearman's rho between
   model rankings under the baseline schedule and each perturbation.

Inputs
------
Per-model CSV files with columns:
    question_id, answer, correct, confidence, delegated, difficulty

Outputs
-------
- reward_sensitivity_summary.csv : per-perturbation rank correlations
- reward_sensitivity_rankings.csv : model rankings under each schedule
- difficulty_sensitivity_summary.csv : rank correlations under threshold changes

Usage
-----
    python compute_sensitivity_analysis.py --input_dir ./results/delegate_game/ \
                                           --output_dir ./results/sensitivity/

Dependencies
------------
    numpy, pandas, scipy
================================================================================
"""

import argparse
import glob
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


# ============================================================================
# Default (Baseline) Reward Schedule
# ============================================================================

BASELINE_REWARD = {
    "correct_conf4": 1.00,
    "correct_conf3": 1.00,
    "correct_conf2": 0.65,
    "correct_conf1": 0.40,
    "incorrect":     0.00,
    "delegate_hard": 0.85,
    "delegate_med":  0.60,
    "delegate_easy": 0.15,
}

BASELINE_THRESHOLDS = {
    "hard_above": 0.65,
    "easy_below": 0.40,
}


# ============================================================================
# Alternative Reward Schedules
# ============================================================================

def generate_reward_perturbations() -> list:
    """
    Generate a set of alternative reward schedules by varying key parameters.

    We perturb three primary dimensions:
    - Delegation reward for hard questions: {0.70, 0.80, 0.85, 0.90, 1.00}
    - Delegation penalty for easy questions: {0.05, 0.10, 0.15, 0.20, 0.30}
    - Confidence bonus spread: narrow (0.60/0.80) vs wide (0.40/0.65) vs flat (0.50/0.50)

    Returns a list of (name, reward_dict) tuples.
    """
    perturbations = []

    # Baseline
    perturbations.append(("baseline", BASELINE_REWARD.copy()))

    # Vary delegation reward for hard questions
    for deleg_hard in [0.70, 0.80, 0.90, 1.00]:
        name = f"deleg_hard_{deleg_hard:.2f}"
        r = BASELINE_REWARD.copy()
        r["delegate_hard"] = deleg_hard
        perturbations.append((name, r))

    # Vary delegation penalty for easy questions
    for deleg_easy in [0.05, 0.10, 0.20, 0.30]:
        name = f"deleg_easy_{deleg_easy:.2f}"
        r = BASELINE_REWARD.copy()
        r["delegate_easy"] = deleg_easy
        perturbations.append((name, r))

    # Vary confidence bonus spread
    # Narrow: less differentiation between confidence levels
    r_narrow = BASELINE_REWARD.copy()
    r_narrow["correct_conf1"] = 0.60
    r_narrow["correct_conf2"] = 0.80
    perturbations.append(("conf_narrow", r_narrow))

    # Flat: no confidence bonus at all
    r_flat = BASELINE_REWARD.copy()
    r_flat["correct_conf1"] = 0.50
    r_flat["correct_conf2"] = 0.50
    r_flat["correct_conf3"] = 0.50
    r_flat["correct_conf4"] = 0.50
    perturbations.append(("conf_flat", r_flat))

    # Combined: high delegation reward + narrow confidence
    r_combo = BASELINE_REWARD.copy()
    r_combo["delegate_hard"] = 0.95
    r_combo["delegate_easy"] = 0.10
    r_combo["correct_conf1"] = 0.55
    r_combo["correct_conf2"] = 0.75
    perturbations.append(("combo_high_deleg_narrow_conf", r_combo))

    return perturbations


def generate_threshold_perturbations() -> list:
    """
    Generate alternative difficulty threshold configurations.

    Returns a list of (name, threshold_dict) tuples.
    """
    perturbations = [
        ("baseline",       {"hard_above": 0.65, "easy_below": 0.40}),
        ("strict_hard",    {"hard_above": 0.75, "easy_below": 0.40}),
        ("lenient_hard",   {"hard_above": 0.55, "easy_below": 0.40}),
        ("strict_easy",    {"hard_above": 0.65, "easy_below": 0.30}),
        ("lenient_easy",   {"hard_above": 0.65, "easy_below": 0.50}),
        ("narrow_medium",  {"hard_above": 0.70, "easy_below": 0.45}),
        ("wide_medium",    {"hard_above": 0.55, "easy_below": 0.35}),
    ]
    return perturbations


# ============================================================================
# Scoring Functions
# ============================================================================

def classify_difficulty(difficulty: float, thresholds: dict) -> str:
    """Classify a question as easy, medium, or hard."""
    if difficulty > thresholds["hard_above"]:
        return "hard"
    elif difficulty < thresholds["easy_below"]:
        return "easy"
    else:
        return "medium"


def score_trial(row: pd.Series, reward: dict, thresholds: dict) -> float:
    """
    Score a single trial under a given reward schedule and threshold set.

    Parameters
    ----------
    row : pd.Series
        Must have columns: correct, confidence, delegated, difficulty.
    reward : dict
        Reward schedule.
    thresholds : dict
        Difficulty thresholds.

    Returns
    -------
    float
        Per-trial score.
    """
    if row["delegated"] == 1:
        diff_class = classify_difficulty(row["difficulty"], thresholds)
        if diff_class == "hard":
            return reward["delegate_hard"]
        elif diff_class == "easy":
            return reward["delegate_easy"]
        else:
            return reward["delegate_med"]
    else:
        if row["correct"] == 1:
            conf = int(row["confidence"]) if not pd.isna(row["confidence"]) else 2
            key = f"correct_conf{conf}"
            return reward.get(key, reward.get("correct_conf2", 0.65))
        else:
            return reward["incorrect"]


def compute_model_score(df: pd.DataFrame, reward: dict,
                        thresholds: dict) -> float:
    """Compute mean per-question reward for a model."""
    scores = df.apply(lambda row: score_trial(row, reward, thresholds), axis=1)
    return scores.mean()


# ============================================================================
# Analysis Pipeline
# ============================================================================

def run_reward_sensitivity(model_dfs: dict, output_dir: str):
    """
    Re-score all models under each reward perturbation and compute
    rank-order correlations against the baseline.
    """
    perturbations = generate_reward_perturbations()
    thresholds = BASELINE_THRESHOLDS

    # Compute scores for each (model, perturbation) pair
    scores = {}  # {perturbation_name: {model_name: score}}
    for pname, reward in perturbations:
        scores[pname] = {}
        for mname, df in model_dfs.items():
            scores[pname][mname] = compute_model_score(df, reward, thresholds)

    # Build rankings DataFrame
    models = sorted(model_dfs.keys())
    rankings_data = []
    for pname, _ in perturbations:
        model_scores = [(m, scores[pname][m]) for m in models]
        model_scores.sort(key=lambda x: -x[1])  # descending by score
        for rank, (m, s) in enumerate(model_scores, 1):
            rankings_data.append({
                "perturbation": pname,
                "model": m,
                "score": s,
                "rank": rank,
            })

    rankings_df = pd.DataFrame(rankings_data)
    rankings_df.to_csv(
        os.path.join(output_dir, "reward_sensitivity_rankings.csv"),
        index=False
    )

    # Compute rank correlations against baseline
    baseline_ranks = rankings_df[rankings_df["perturbation"] == "baseline"] \
        .set_index("model")["rank"]

    summary = []
    for pname, _ in perturbations:
        if pname == "baseline":
            continue
        perturbed_ranks = rankings_df[rankings_df["perturbation"] == pname] \
            .set_index("model")["rank"]

        # Align on model names
        common = baseline_ranks.index.intersection(perturbed_ranks.index)
        if len(common) < 3:
            continue

        tau, tau_p = kendalltau(baseline_ranks[common], perturbed_ranks[common])
        rho, rho_p = spearmanr(baseline_ranks[common], perturbed_ranks[common])

        summary.append({
            "perturbation": pname,
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "n_models": len(common),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        os.path.join(output_dir, "reward_sensitivity_summary.csv"),
        index=False
    )

    print("=== REWARD SENSITIVITY ===")
    print(summary_df.to_string(index=False))
    print()

    return summary_df


def run_threshold_sensitivity(model_dfs: dict, output_dir: str):
    """
    Re-score all models under each difficulty-threshold perturbation
    and compute rank-order correlations against the baseline.
    """
    perturbations = generate_threshold_perturbations()
    reward = BASELINE_REWARD

    models = sorted(model_dfs.keys())
    scores = {}
    for tname, thresholds in perturbations:
        scores[tname] = {}
        for mname, df in model_dfs.items():
            scores[tname][mname] = compute_model_score(df, reward, thresholds)

    # Build rankings
    rankings_data = []
    for tname, _ in perturbations:
        model_scores = [(m, scores[tname][m]) for m in models]
        model_scores.sort(key=lambda x: -x[1])
        for rank, (m, s) in enumerate(model_scores, 1):
            rankings_data.append({
                "threshold_config": tname,
                "model": m,
                "score": s,
                "rank": rank,
            })

    rankings_df = pd.DataFrame(rankings_data)
    rankings_df.to_csv(
        os.path.join(output_dir, "difficulty_sensitivity_rankings.csv"),
        index=False
    )

    # Rank correlations
    baseline_ranks = rankings_df[rankings_df["threshold_config"] == "baseline"] \
        .set_index("model")["rank"]

    summary = []
    for tname, _ in perturbations:
        if tname == "baseline":
            continue
        perturbed_ranks = rankings_df[rankings_df["threshold_config"] == tname] \
            .set_index("model")["rank"]
        common = baseline_ranks.index.intersection(perturbed_ranks.index)
        if len(common) < 3:
            continue

        tau, tau_p = kendalltau(baseline_ranks[common], perturbed_ranks[common])
        rho, rho_p = spearmanr(baseline_ranks[common], perturbed_ranks[common])

        summary.append({
            "threshold_config": tname,
            "kendall_tau": tau,
            "kendall_p": tau_p,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "n_models": len(common),
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(
        os.path.join(output_dir, "difficulty_sensitivity_summary.csv"),
        index=False
    )

    print("=== DIFFICULTY-THRESHOLD SENSITIVITY ===")
    print(summary_df.to_string(index=False))
    print()

    return summary_df


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Reward-sensitivity and difficulty-threshold sensitivity "
                    "analysis for the Delegate Game."
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default="./results/sensitivity/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        print(f"ERROR: No CSV files found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    # Load all models
    model_dfs = {}
    for fpath in csv_files:
        mname = os.path.splitext(os.path.basename(fpath))[0]
        try:
            model_dfs[mname] = pd.read_csv(fpath)
        except Exception as e:
            print(f"WARNING: Could not load {fpath}: {e}", file=sys.stderr)

    print(f"Loaded {len(model_dfs)} model(s)\n")

    run_reward_sensitivity(model_dfs, args.output_dir)
    run_threshold_sensitivity(model_dfs, args.output_dir)

    print("Done. All outputs saved to:", args.output_dir)


if __name__ == "__main__":
    main()
