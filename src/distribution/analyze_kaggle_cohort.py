#!/usr/bin/env python3
"""
analyze_kaggle_cohort.py
========================

Runs the full Kaggle-cohort analysis against the extraction produced by
`extract_kaggle_outputs.py`. Mirrors the Turing-cohort analysis pipeline
(comparative baselines, sensitivity, ranking stability, cross-task
dissociations) but adapted for the 23-model × 11-task Kaggle data.

Usage:
    python analyze_kaggle_cohort.py \
        --extracted-dir ./kaggle_extracted \
        --output-dir ./kaggle_analysis

Produces (under `--output-dir`):
    comparative/
        task_scores_matrix.csv        (23 models × 11 tasks, platform score from run.json)
        delegate_game_metrics.csv     (per-model Task 1 metrics: deleg rate, AUC, ECE, Brier, etc.)
        rank_reversal_pairs.csv       (pairs of models whose rankings flip across task pairs)
    sensitivity/
        task1_reward_sensitivity.csv  (11 reward-schedule perturbations on Task 1)
        task1_difficulty_sensitivity.csv (6 difficulty-threshold perturbations)
    stability/
        task1_ranking_stability.csv   (1000 random item-subsampling splits)
        task1_item_discrimination.csv (per-question r_pb for all Task 1 items)
    dissociations/
        cross_task_spearman.csv       (pairwise Spearman rho between tasks)
        per_model_profile.csv         (each model's score vector + dissociation summary)
        dissociation_highlights.csv   (model-pairs with largest rank divergence)
    summary_findings.md               (human-readable summary of the headline findings)

Notes on scoring convention:
    Two "mean scores" exist per run. This analysis uses the platform-
    authoritative score (`run_result_value` from run_metadata.csv, which
    counts missing trials as 0) for overall task rankings, because that
    is what the Kaggle leaderboard publishes and what reviewers checking
    our work will see.

    For per-question analyses (rank reversals at the trial level,
    sensitivity to reward perturbations), we use only the completed
    trials in `per_task/*.csv`, and we explicitly note this in every
    output file.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

TASK_IDS = ["t01", "t02", "t03", "t04", "t05",
            "t06", "t07", "t08", "t09", "t10", "t11"]

TASK_SHORT_NAMES = {
    "t01": "delegate_game",      "t02": "declared_probe",
    "t03": "second_chance",      "t04": "confidence_entropy",
    "t05": "teammate_delegate",  "t06": "behavioral_er",
    "t07": "behavioral_ci",      "t08": "behavioral_em",
    "t09": "behavioral_pi",      "t10": "dpp_sequence",
    "t11": "mc_binary_pairs",
}

TASK_HUMAN_NAMES = {
    "t01": "Delegate Game",          "t02": "Declared MSV Probe",
    "t03": "Second-Chance Game",     "t04": "Confidence Entropy",
    "t05": "Teammate Delegate",      "t06": "Behavioral ER",
    "t07": "Behavioral CI",          "t08": "Behavioral EM",
    "t09": "Behavioral PI",          "t10": "DPP Sequence",
    "t11": "MC Binary Pairs",
}


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def load_task_scores_matrix(run_metadata_path: Path) -> pd.DataFrame:
    """Return a wide DataFrame: models × tasks, cells are platform scores."""
    md = pd.read_csv(run_metadata_path)
    md["run_result_value"] = pd.to_numeric(md["run_result_value"], errors="coerce")
    wide = md.pivot_table(
        index="model", columns="task_id",
        values="run_result_value", aggfunc="first",
    )
    wide = wide.reindex(columns=TASK_IDS)
    wide.columns = [f"{t}_{TASK_SHORT_NAMES[t]}" for t in wide.columns]
    return wide


def load_task_long(extracted_dir: Path, task_id: str) -> pd.DataFrame:
    """Load the long-form per-task CSV for one task."""
    short = TASK_SHORT_NAMES[task_id]
    path = extracted_dir / "per_task" / f"{task_id}_{short}.csv"
    return pd.read_csv(path)


# --------------------------------------------------------------------------
# Delegate Game (Task 1) metrics: deleg rate, AUC, MCC, NRB-like, ECE/Brier
# --------------------------------------------------------------------------

def _ece(confidence_01, correct_binary, n_bins=4):
    """Expected Calibration Error grouped by discrete confidence label.

    Per paper Section 4.3: ECE is computed with one bin per discrete
    confidence level (k in {1, 2, 3, 4}), not by equal-width binning of
    the [0, 1] interval. This avoids boundary ambiguity (which would
    silently merge confidence levels 3 and 4 under the canonical
    mapping {0.25, 0.50, 0.75, 1.00}) and matches the paper's stated
    estimator. The n_bins parameter is retained for backward
    compatibility but unused: the binning is always over the four
    discrete confidence labels.

    confidence_01 is expected to be (ans_conf / 4.0) so that the
    discrete labels {1, 2, 3, 4} map to {0.25, 0.50, 0.75, 1.00}.
    The original integer label is recovered by rounding 4 * conf_01.
    """
    del n_bins  # retained in signature for API compatibility; unused
    if len(confidence_01) == 0:
        return np.nan
    conf01 = np.asarray(confidence_01, dtype=float)
    corr = np.asarray(correct_binary, dtype=float)
    # Recover integer confidence labels from the normalized values.
    # ans_conf / 4 -> label = round(conf01 * 4)
    labels = np.rint(conf01 * 4).astype(int)
    n = len(conf01)
    ece = 0.0
    for k in (1, 2, 3, 4):
        mask = (labels == k)
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_k = k / 4.0
        acc_k = corr[mask].mean()
        ece += (nb / n) * abs(acc_k - p_k)
    return ece


def _brier(confidence_01, correct_binary):
    if len(confidence_01) == 0:
        return np.nan
    return float(np.mean((np.asarray(confidence_01) - np.asarray(correct_binary)) ** 2))


def compute_delegate_game_metrics(task1_long: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model behavioral + declarative metrics from Task 1 trials.

    Task 1 per-trial columns:
        model, question_id, choice, answer, confidence, correct, difficulty,
        score, raw_response
    choice is ANSWER or DELEGATE. correct is the letter A/B/C/D, answer is
    what the model chose (when ANSWER). confidence is 1-4, difficulty in
    [0,1].
    """
    results = []
    for model, grp in task1_long.groupby("model"):
        n = len(grp)
        if n == 0:
            continue
        answered = grp["choice"] == "ANSWER"
        delegated = grp["choice"] == "DELEGATE"
        # Did the model answer correctly (only meaningful on ANSWER trials)
        ans = grp[answered].copy()
        if len(ans) > 0:
            ans_correct = (ans["answer"] == ans["correct"]).astype(int)
            ans_conf = pd.to_numeric(ans["confidence"], errors="coerce").fillna(0)
            conf_01 = ans_conf / 4.0          # 1-4 -> 0.25, 0.50, 0.75, 1.00 (paper Section 4.3 canonical mapping)
            # ECE and Brier on answered trials (standard calibration metrics)
            ece = _ece(conf_01, ans_correct)
            brier = _brier(conf_01, ans_correct)
            ans_accuracy = ans_correct.mean()
            ans_mean_conf = ans_conf.mean()
        else:
            ece = brier = ans_accuracy = ans_mean_conf = np.nan

        # Delegation AUC-ROC: does the model delegate *away from* items it
        # would get wrong? We form a binary label per trial: 1 if the
        # model's underlying proficiency would score poorly, 0 otherwise.
        # We can only observe this on ANSWER trials, so we use the observed
        # incorrectness as the target and the (1 - delegation-decision) as
        # the "I will attempt" signal. Equivalent: AUC(delegate, wrong|answered)
        # extended to full cohort by treating DELEGATE as a positive signal
        # against empirical difficulty.
        #
        # We use the simpler, directly-meaningful formulation:
        # among items with known difficulty, rank by P(delegate) and see
        # whether that ranking discriminates harder-than-median items.
        # NOTE: the median here is per-model (computed within this model's
        # data subset). This is a less defensible threshold than the global
        # panel median used in compute_rank_divergence_ci.py with
        # --auc_target=hardness, which computes the median once across the
        # union of unique (question_id, difficulty) pairs and is a stable
        # item-level property. The paper's appendix audit reports the
        # global-median version; the per-model column produced here is
        # retained for backward compatibility and audit history but is NOT
        # the column referenced in any paper-bound result. See the
        # `deleg_auc_vs_hardness_per_model_median` column in this script's
        # output and the `compute_delegation_auc_vs_hardness` function in
        # compute_rank_divergence_ci.py for the global-median form.
        difficulty = pd.to_numeric(grp["difficulty"], errors="coerce")
        median_diff = difficulty.median()  # per-model median
        hard_label = (difficulty >= median_diff).astype(int).values
        delegate_sig = delegated.astype(int).values
        if len(set(hard_label)) > 1 and len(set(delegate_sig)) > 1:
            try:
                deleg_auc_vs_hardness_per_model_median = roc_auc_score(hard_label, delegate_sig)
            except Exception:
                deleg_auc_vs_hardness_per_model_median = np.nan
        else:
            deleg_auc_vs_hardness_per_model_median = np.nan

        # Delegation AUC against the model's OWN error probability.
        # This is the paper's primary behavioral metric (see Appendix G).
        # Label = 1 if the model would be incorrect if it attempted; we
        # observe "would be incorrect" only on ANSWER trials, but the AUC
        # is computed over all trials by treating (answer == correct) as
        # correctness on ANSWER trials and using the model's delegation
        # decision as the score. Delegated trials contribute to the
        # cohort but their correctness is unobserved; scikit-learn's
        # roc_auc_score handles NaN labels by excluding them. We adopt
        # the more natural formulation: per-trial correctness is 1 if
        # the model answered correctly, 0 if it answered incorrectly or
        # delegated (delegation is treated as "did not get this right").
        # This matches rank_divergence_audit.csv.
        own_correct = np.where(
            grp["choice"].values == "ANSWER",
            (grp["answer"].values == grp["correct"].values).astype(int),
            0,  # delegation counts as "did not answer correctly"
        )
        own_error_label = 1 - own_correct  # 1 = incorrect or delegated
        if len(set(own_error_label)) > 1 and len(set(delegate_sig)) > 1:
            try:
                deleg_auc_vs_own_err = roc_auc_score(own_error_label, delegate_sig)
            except Exception:
                deleg_auc_vs_own_err = np.nan
        else:
            deleg_auc_vs_own_err = np.nan

        results.append({
            "model":                 model,
            "n_trials":              n,
            "delegation_rate":       delegated.mean(),
            "answer_rate":           answered.mean(),
            "ans_accuracy":          ans_accuracy,
            "ans_mean_conf":         ans_mean_conf,
            "ece_4bin":                              ece,
            "brier":                                 brier,
            "deleg_auc_vs_hardness_per_model_median": deleg_auc_vs_hardness_per_model_median,
            "deleg_auc_vs_own_err":                  deleg_auc_vs_own_err,
            "mean_score":            grp["score"].astype(float).mean(),
        })
    return pd.DataFrame(results).sort_values("mean_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Sensitivity: reward-schedule and difficulty-threshold perturbations
# --------------------------------------------------------------------------

REWARD_PERTURBATIONS = [
    # (name, hard_delegate_reward, easy_delegate_penalty, conf_bonus_weight)
    ("baseline",             0.85, 0.10, 0.10),
    ("deleg_hard_0.70",      0.70, 0.10, 0.10),
    ("deleg_hard_0.80",      0.80, 0.10, 0.10),
    ("deleg_hard_0.90",      0.90, 0.10, 0.10),
    ("deleg_hard_1.00",      1.00, 0.10, 0.10),
    ("deleg_easy_0.05",      0.85, 0.05, 0.10),
    ("deleg_easy_0.10",      0.85, 0.10, 0.10),
    ("deleg_easy_0.20",      0.85, 0.20, 0.10),
    ("deleg_easy_0.30",      0.85, 0.30, 0.10),
    ("conf_narrow",          0.85, 0.10, 0.20),
    ("conf_flat",            0.85, 0.10, 0.00),
    ("combo_high_deleg_narrow_conf", 0.95, 0.05, 0.20),
]


def rescore_task1(task1_long: pd.DataFrame, hard_del, easy_pen, conf_w,
                  difficulty_threshold=0.6):
    """Recompute per-trial scores with a different reward schedule.

    Scoring logic (mirrors the Delegate Game scoring in the Kaggle notebook):
        - If model ANSWER and answer is correct: 1.0 + conf_w * (conf - 2)/2
        - If model ANSWER and answer is incorrect: 0.0 + conf_w * (2 - conf)/2
        - If model DELEGATE on a hard item (difficulty >= threshold): hard_del
        - If model DELEGATE on an easy item (difficulty < threshold): -easy_pen
    Then the model-level score is the mean across trials.
    """
    df = task1_long.copy()
    df["difficulty_num"] = pd.to_numeric(df["difficulty"], errors="coerce")
    df["conf_num"] = pd.to_numeric(df["confidence"], errors="coerce").fillna(2)
    scores = np.zeros(len(df))
    is_answer = df["choice"].values == "ANSWER"
    is_delegate = df["choice"].values == "DELEGATE"
    is_correct = (df["answer"].values == df["correct"].values)
    is_hard = df["difficulty_num"].values >= difficulty_threshold
    conf_shift = (df["conf_num"].values - 2) / 2.0
    # Answered correctly
    scores[is_answer & is_correct] = 1.0 + conf_w * conf_shift[is_answer & is_correct]
    # Answered incorrectly
    scores[is_answer & ~is_correct] = 0.0 - conf_w * conf_shift[is_answer & ~is_correct]
    # Delegated
    scores[is_delegate & is_hard]  = hard_del
    scores[is_delegate & ~is_hard] = -easy_pen
    df["perturbed_score"] = scores
    return df.groupby("model")["perturbed_score"].mean()


def compute_reward_sensitivity(task1_long: pd.DataFrame) -> pd.DataFrame:
    """For each reward perturbation, rank models and compare to baseline ranking."""
    baseline_scores = rescore_task1(task1_long, *REWARD_PERTURBATIONS[0][1:])
    baseline_rank = baseline_scores.rank(ascending=False)

    rows = []
    for name, hd, ep, cw in REWARD_PERTURBATIONS:
        scores = rescore_task1(task1_long, hd, ep, cw)
        rnk = scores.rank(ascending=False)
        # Intersect models (they should match)
        common = baseline_rank.index.intersection(rnk.index)
        if len(common) < 3:
            continue
        tau, _ = stats.kendalltau(baseline_rank.loc[common], rnk.loc[common])
        rho, _ = stats.spearmanr(baseline_rank.loc[common], rnk.loc[common])
        rows.append({
            "perturbation":    name,
            "hard_del_reward": hd,
            "easy_del_penalty": ep,
            "conf_bonus_weight": cw,
            "n_models":        len(common),
            "kendall_tau_vs_baseline": tau,
            "spearman_rho_vs_baseline": rho,
        })
    return pd.DataFrame(rows)


DIFFICULTY_PERTURBATIONS = [
    ("baseline",       0.60),
    ("strict_hard",    0.75),
    ("lenient_hard",   0.55),
    ("strict_easy",    0.30),
    ("lenient_easy",   0.50),
    ("narrow_medium",  0.45),
    ("wide_medium",    0.65),
]


def compute_difficulty_sensitivity(task1_long: pd.DataFrame) -> pd.DataFrame:
    base_thr = DIFFICULTY_PERTURBATIONS[0][1]
    baseline = rescore_task1(task1_long, 0.85, 0.10, 0.10, difficulty_threshold=base_thr)
    baseline_rank = baseline.rank(ascending=False)
    rows = []
    for name, thr in DIFFICULTY_PERTURBATIONS:
        scores = rescore_task1(task1_long, 0.85, 0.10, 0.10, difficulty_threshold=thr)
        rnk = scores.rank(ascending=False)
        common = baseline_rank.index.intersection(rnk.index)
        if len(common) < 3:
            continue
        tau, _ = stats.kendalltau(baseline_rank.loc[common], rnk.loc[common])
        rho, _ = stats.spearmanr(baseline_rank.loc[common], rnk.loc[common])
        rows.append({
            "perturbation":     name,
            "difficulty_threshold": thr,
            "n_models":         len(common),
            "kendall_tau_vs_baseline": tau,
            "spearman_rho_vs_baseline": rho,
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Ranking stability via random item-subsampling
# --------------------------------------------------------------------------

def compute_ranking_stability(task1_long: pd.DataFrame, n_splits: int = 1000,
                              seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """For each of n_splits random partitions of the question set into halves,
    rank models on each half and compute correlation between the two rankings.

    Returns (summary, per_split).
    """
    rng = np.random.default_rng(seed)
    # Only use questions that all models attempted (avoid subsampling bias)
    counts = task1_long.groupby("question_id")["model"].nunique()
    n_models = task1_long["model"].nunique()
    common_qs = counts[counts == n_models].index.tolist()
    if len(common_qs) < 10:
        # Fall back to questions with at least half the models
        common_qs = counts[counts >= n_models // 2].index.tolist()
    common_df = task1_long[task1_long["question_id"].isin(common_qs)].copy()

    per_split = []
    for i in range(n_splits):
        shuffled = rng.permutation(common_qs)
        half = len(shuffled) // 2
        a_qs = set(shuffled[:half])
        b_qs = set(shuffled[half:2*half])
        a_scores = common_df[common_df["question_id"].isin(a_qs)].groupby("model")["score"].mean()
        b_scores = common_df[common_df["question_id"].isin(b_qs)].groupby("model")["score"].mean()
        common = a_scores.index.intersection(b_scores.index)
        if len(common) < 3:
            continue
        rho, _ = stats.spearmanr(a_scores.loc[common], b_scores.loc[common])
        tau, _ = stats.kendalltau(a_scores.loc[common].rank(), b_scores.loc[common].rank())
        r, _ = stats.pearsonr(a_scores.loc[common], b_scores.loc[common])
        per_split.append({"split_idx": i, "spearman_rho": rho, "kendall_tau": tau, "pearson_r": r})

    ps = pd.DataFrame(per_split)
    sb = 2 * ps["spearman_rho"].mean() / (1 + ps["spearman_rho"].mean())
    summary = pd.DataFrame([{
        "n_splits":           len(ps),
        "n_common_questions": len(common_qs),
        "n_models":           n_models,
        "mean_spearman_rho":  ps["spearman_rho"].mean(),
        "mean_kendall_tau":   ps["kendall_tau"].mean(),
        "mean_pearson_r":     ps["pearson_r"].mean(),
        "std_spearman_rho":   ps["spearman_rho"].std(),
        "std_kendall_tau":    ps["kendall_tau"].std(),
        "spearman_brown_corrected": sb,
    }])
    return summary, ps


def compute_item_discrimination(task1_long: pd.DataFrame) -> pd.DataFrame:
    """Per-question r_pb between delegation decision and incorrectness.

    Only includes questions with enough variance (both DELEGATE and ANSWER
    represented among the models).
    """
    rows = []
    for qid, grp in task1_long.groupby("question_id"):
        if len(grp) < 3:
            continue
        delegate = (grp["choice"] == "DELEGATE").astype(int).values
        answered_correctly = ((grp["choice"] == "ANSWER") &
                              (grp["answer"] == grp["correct"])).astype(int).values
        # For r_pb: the binary variable is "wrong" (0 if right or correctly
        # delegated on difficult item; 1 if wrong or delegated on easy item).
        # Simpler: use "delegate" as binary var and item mean score as
        # continuous var.
        score = pd.to_numeric(grp["score"], errors="coerce").values
        if len(set(delegate)) > 1 and np.std(score) > 1e-9:
            r_pb, p = stats.pointbiserialr(delegate, score)
        else:
            r_pb, p = (np.nan, np.nan)
        rows.append({
            "question_id":         qid,
            "n_models":            len(grp),
            "delegation_rate":     delegate.mean(),
            "mean_score":          score.mean(),
            "discrimination_rpb":  r_pb,
            "discrimination_p":    p,
        })
    return pd.DataFrame(rows).sort_values("question_id").reset_index(drop=True)


# --------------------------------------------------------------------------
# Cross-task dissociations
# --------------------------------------------------------------------------

def compute_cross_task_correlations(scores_matrix: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Spearman rho between tasks (over models)."""
    rows = []
    cols = list(scores_matrix.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            valid = scores_matrix[[a, b]].dropna()
            if len(valid) < 3:
                continue
            rho, p = stats.spearmanr(valid[a], valid[b])
            tau, _ = stats.kendalltau(valid[a].rank(), valid[b].rank())
            rows.append({
                "task_a":       a,
                "task_b":       b,
                "n_models":     len(valid),
                "spearman_rho": rho,
                "spearman_p":   p,
                "kendall_tau":  tau,
            })
    return pd.DataFrame(rows).sort_values("spearman_rho").reset_index(drop=True)


def compute_dissociation_highlights(scores_matrix: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """For each model, compute its task-score vector and flag the biggest
    within-model swings (where rank on one task differs dramatically from rank on another)."""
    # Rank each task (over models)
    ranks = scores_matrix.rank(ascending=False, method="min")
    rows = []
    tasks = list(scores_matrix.columns)
    for model in scores_matrix.index:
        model_ranks = ranks.loc[model].dropna()
        if len(model_ranks) < 2:
            continue
        rng = model_ranks.max() - model_ranks.min()
        best_task = model_ranks.idxmin()
        worst_task = model_ranks.idxmax()
        rows.append({
            "model":         model,
            "rank_range":    rng,
            "best_task":     best_task,
            "best_rank":     int(model_ranks.min()),
            "worst_task":    worst_task,
            "worst_rank":    int(model_ranks.max()),
            "mean_rank":     model_ranks.mean(),
            "std_rank":      model_ranks.std(),
            "n_tasks":       len(model_ranks),
        })
    return pd.DataFrame(rows).sort_values("rank_range", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Summary writer
# --------------------------------------------------------------------------

def write_summary_findings(out_path: Path, scores: pd.DataFrame, dg_metrics: pd.DataFrame,
                           reward_sens: pd.DataFrame, diff_sens: pd.DataFrame,
                           stability_summary: pd.DataFrame,
                           item_disc: pd.DataFrame, xtask_corr: pd.DataFrame,
                           dissoc: pd.DataFrame):
    """Write a human-readable markdown summary of the headline findings."""
    lines = ["# Kaggle Cohort Analysis — Summary Findings", ""]
    lines.append("This file is auto-generated by `analyze_kaggle_cohort.py`.")
    lines.append("All numbers are derived from the 23-model × 11-task Kaggle run archive.")
    lines.append("")

    # Cohort summary
    lines.append("## Cohort")
    lines.append(f"- Models: {len(scores)} (full set of Kaggle-hosted frontier + mid-tier)")
    lines.append(f"- Tasks: {len(scores.columns)}")
    coverage = scores.notna().sum(axis=1)
    lines.append(f"- Models with complete coverage (all 11 tasks): "
                 f"{(coverage == len(scores.columns)).sum()} / {len(scores)}")
    lines.append("")

    # Overall leaderboard (top + bottom)
    lines.append("## Overall Leaderboard (mean of platform scores across 11 tasks)")
    overall = scores.mean(axis=1).sort_values(ascending=False)
    lines.append("| Rank | Model | Mean Score | Tasks Scored |")
    lines.append("|------|-------|-----------|--------------|")
    for i, (m, s) in enumerate(overall.head(10).items(), 1):
        n = scores.loc[m].notna().sum()
        lines.append(f"| {i} | `{m}` | {s:.4f} | {n}/11 |")
    lines.append("")

    # Delegate Game (Task 1) comparative metrics
    lines.append("## Delegate Game (Task 1) Per-Model Metrics")
    lines.append("Computed on completed trials only. `ece_4bin` and `brier` are declarative")
    lines.append("calibration metrics; `deleg_auc_vs_own_err` is the paper's primary")
    lines.append("behavioral metric (delegation tracks the model's own error probability);")
    lines.append("`deleg_auc_vs_hardness_per_model_median` is an audit-only column using")
    lines.append("a per-model difficulty median. The paper's appendix reports the")
    lines.append("global-median form computed by compute_rank_divergence_ci.py with")
    lines.append("--auc_target=hardness. See Appendix G of the paper.")
    lines.append("")
    lines.append("| Model | N Trials | Deleg Rate | Ans Accuracy | ECE | Brier | Deleg AUC (own err) | Deleg AUC (hardness, per-model median) | Mean Score |")
    lines.append("|-------|----------|------------|--------------|-----|-------|---------------------|----------------------|------------|")
    for _, r in dg_metrics.iterrows():
        def fmt(v):
            return f"{v:.2f}" if pd.notna(v) else "-"
        lines.append(
            f"| `{r['model']}` | {int(r['n_trials'])} | {r['delegation_rate']:.2f} | "
            f"{fmt(r['ans_accuracy'])} | {fmt(r['ece_4bin'])} | {fmt(r['brier'])} | "
            f"{fmt(r['deleg_auc_vs_own_err'])} | {fmt(r['deleg_auc_vs_hardness_per_model_median'])} | "
            f"{r['mean_score']:.3f} |"
        )
    lines.append("")

    # Rank reversals
    lines.append("## Rank Reversals (Behavioral vs. Declarative)")
    lines.append("Primary analysis: models ranked by declarative ECE (best = lowest) vs")
    lines.append("behavioral Delegation AUC against own error (best = highest). Divergence")
    lines.append("in these rankings is the paper's central finding. See Appendix G for the")
    lines.append("full sensitivity grid including hardness AUC comparisons.")
    lines.append("")
    # Primary: own-error AUC, all-computable (paper's primary estimand)
    ece_rank = dg_metrics.set_index("model")["ece_4bin"].rank(ascending=True)
    auc_rank_own = dg_metrics.set_index("model")["deleg_auc_vs_own_err"].rank(ascending=False)
    cmp_df = pd.DataFrame({"ECE_rank": ece_rank, "DelegAUC_rank": auc_rank_own}).dropna()
    cmp_df["rank_diff"] = (cmp_df["ECE_rank"] - cmp_df["DelegAUC_rank"]).abs()
    tau, p = stats.kendalltau(cmp_df["ECE_rank"], cmp_df["DelegAUC_rank"])
    rho, _ = stats.spearmanr(cmp_df["ECE_rank"], cmp_df["DelegAUC_rank"])
    lines.append(f"**Primary (own-error AUC, all-computable, n={len(cmp_df)}):**")
    lines.append(f"- Kendall tau: **{tau:+.3f}** (p={p:.3f})")
    lines.append(f"- Spearman rho: **{rho:+.3f}**")
    lines.append("- Top rank-divergent models (|ECE_rank - DelegAUC_rank|):")
    cmp_df_sorted = cmp_df.sort_values("rank_diff", ascending=False).head(5)
    for m, r in cmp_df_sorted.iterrows():
        lines.append(f"    - `{m}`: ECE rank {int(r['ECE_rank'])}, DelegAUC rank {int(r['DelegAUC_rank'])}, diff = {int(r['rank_diff'])}")
    lines.append("")
    # Sensitivity: hardness AUC, all-computable (the original hardness-AUC estimand)
    auc_rank_hard = dg_metrics.set_index("model")["deleg_auc_vs_hardness_per_model_median"].rank(ascending=False)
    cmp_df_h = pd.DataFrame({"ECE_rank": ece_rank, "DelegAUC_rank": auc_rank_hard}).dropna()
    tau_h, p_h = stats.kendalltau(cmp_df_h["ECE_rank"], cmp_df_h["DelegAUC_rank"])
    rho_h, _ = stats.spearmanr(cmp_df_h["ECE_rank"], cmp_df_h["DelegAUC_rank"])
    lines.append(f"**Sensitivity (hardness AUC, all-computable, n={len(cmp_df_h)}):**")
    lines.append(f"- Kendall tau: **{tau_h:+.3f}** (p={p_h:.3f})")
    lines.append(f"- Spearman rho: **{rho_h:+.3f}**")
    lines.append("")

    # Sensitivity
    lines.append("## Sensitivity Analysis (Task 1)")
    lines.append("### Reward-Schedule Perturbations (11 configurations)")
    lines.append("Kendall tau and Spearman rho between each perturbed ranking and baseline.")
    lines.append("")
    lines.append("| Perturbation | Kendall tau | Spearman rho |")
    lines.append("|--------------|-------------|--------------|")
    for _, r in reward_sens.iterrows():
        lines.append(f"| {r['perturbation']} | {r['kendall_tau_vs_baseline']:.3f} | {r['spearman_rho_vs_baseline']:.3f} |")
    tau_min = reward_sens["kendall_tau_vs_baseline"].min()
    lines.append("")
    lines.append(f"**Min Kendall tau across perturbations: {tau_min:.3f}**")
    lines.append("")

    lines.append("### Difficulty-Threshold Perturbations (7 configurations)")
    lines.append("| Perturbation | Threshold | Kendall tau | Spearman rho |")
    lines.append("|--------------|-----------|-------------|--------------|")
    for _, r in diff_sens.iterrows():
        lines.append(f"| {r['perturbation']} | {r['difficulty_threshold']:.2f} | "
                     f"{r['kendall_tau_vs_baseline']:.3f} | {r['spearman_rho_vs_baseline']:.3f} |")
    lines.append("")

    # Stability
    lines.append("## Ranking Stability via Item Subsampling")
    if len(stability_summary) > 0:
        s = stability_summary.iloc[0]
        lines.append(f"- Splits: {int(s['n_splits'])} random halvings of {int(s['n_common_questions'])} items common across {int(s['n_models'])} models")
        lines.append(f"- Mean Spearman rho: **{s['mean_spearman_rho']:.3f}** (std {s['std_spearman_rho']:.3f})")
        lines.append(f"- Mean Kendall tau: **{s['mean_kendall_tau']:.3f}** (std {s['std_kendall_tau']:.3f})")
        lines.append(f"- Mean Pearson r: {s['mean_pearson_r']:.3f}")
        lines.append(f"- Spearman-Brown corrected: {s['spearman_brown_corrected']:.3f}")
    lines.append("")

    # Item discrimination
    if len(item_disc) > 0:
        usable = item_disc.dropna(subset=["discrimination_rpb"])
        good = (usable["discrimination_rpb"].abs() > 0.3).sum()
        mean_rpb = usable["discrimination_rpb"].mean()
        lines.append("## Item Discrimination")
        lines.append(f"- Questions with computable r_pb: {len(usable)} of {len(item_disc)}")
        lines.append(f"- Questions with |r_pb| > 0.3 (good discrimination): {good} / {len(usable)}")
        lines.append(f"- Mean r_pb: {mean_rpb:.3f}")
        lines.append("")

    # Cross-task correlations
    lines.append("## Cross-Task Correlations")
    lines.append("Pairwise Spearman rho between task rankings (over models). Low correlations")
    lines.append("indicate that a good ranking on one task does not predict a good ranking on another.")
    lines.append("")
    lines.append("### Lowest-correlation task pairs")
    lines.append("| Task A | Task B | N Models | Spearman rho |")
    lines.append("|--------|--------|----------|--------------|")
    for _, r in xtask_corr.head(8).iterrows():
        lines.append(f"| {r['task_a']} | {r['task_b']} | {int(r['n_models'])} | {r['spearman_rho']:.3f} |")
    lines.append("")
    lines.append("### Highest-correlation task pairs")
    lines.append("| Task A | Task B | N Models | Spearman rho |")
    lines.append("|--------|--------|----------|--------------|")
    for _, r in xtask_corr.tail(8)[::-1].iterrows():
        lines.append(f"| {r['task_a']} | {r['task_b']} | {int(r['n_models'])} | {r['spearman_rho']:.3f} |")
    lines.append("")

    # Dissociation highlights
    lines.append("## Model-Level Dissociation Highlights")
    lines.append("Models sorted by rank-range (max rank - min rank across tasks). Large ranges")
    lines.append("indicate within-model dissociation: the model is strong on some dimensions but")
    lines.append("weak on others.")
    lines.append("")
    lines.append("| Model | Rank Range | Best Task (rank) | Worst Task (rank) | Mean Rank | Std Rank |")
    lines.append("|-------|-----------|------------------|-------------------|-----------|----------|")
    for _, r in dissoc.head(12).iterrows():
        lines.append(f"| `{r['model']}` | {int(r['rank_range'])} | {r['best_task']} ({r['best_rank']}) | "
                     f"{r['worst_task']} ({r['worst_rank']}) | {r['mean_rank']:.1f} | {r['std_rank']:.1f} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("End of auto-generated summary.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracted-dir", required=True, type=Path)
    ap.add_argument("--output-dir",    required=True, type=Path)
    ap.add_argument("--n-splits",      type=int, default=1000)
    ap.add_argument("--seed",          type=int, default=42)
    args = ap.parse_args()

    out = args.output_dir
    (out / "comparative").mkdir(parents=True, exist_ok=True)
    (out / "sensitivity").mkdir(parents=True, exist_ok=True)
    (out / "stability").mkdir(parents=True, exist_ok=True)
    (out / "dissociations").mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {args.extracted_dir}...")
    scores = load_task_scores_matrix(args.extracted_dir / "run_metadata.csv")
    scores.to_csv(out / "comparative" / "task_scores_matrix.csv")
    print(f"  task_scores_matrix: {scores.shape[0]} models × {scores.shape[1]} tasks")

    task1 = load_task_long(args.extracted_dir, "t01")
    print(f"  Task 1 long form: {len(task1)} trials across {task1['model'].nunique()} models")

    print("\n[comparative] Computing Delegate Game metrics...")
    dg = compute_delegate_game_metrics(task1)
    dg.to_csv(out / "comparative" / "delegate_game_metrics.csv", index=False)
    print(f"  delegate_game_metrics: {len(dg)} rows")

    print("\n[sensitivity] Reward-schedule perturbations...")
    reward_sens = compute_reward_sensitivity(task1)
    reward_sens.to_csv(out / "sensitivity" / "task1_reward_sensitivity.csv", index=False)
    print(f"  task1_reward_sensitivity: {len(reward_sens)} perturbations")

    print("\n[sensitivity] Difficulty-threshold perturbations...")
    diff_sens = compute_difficulty_sensitivity(task1)
    diff_sens.to_csv(out / "sensitivity" / "task1_difficulty_sensitivity.csv", index=False)
    print(f"  task1_difficulty_sensitivity: {len(diff_sens)} perturbations")

    print(f"\n[stability] {args.n_splits} random subsampling splits...")
    stab_summary, stab_dist = compute_ranking_stability(task1, n_splits=args.n_splits, seed=args.seed)
    stab_summary.to_csv(out / "stability" / "task1_ranking_stability_summary.csv", index=False)
    stab_dist.to_csv(out / "stability" / "task1_ranking_stability_distribution.csv", index=False)
    print(f"  ranking_stability: mean rho = {stab_summary['mean_spearman_rho'].iloc[0]:.3f}")

    print("\n[stability] Item discrimination...")
    item_disc = compute_item_discrimination(task1)
    item_disc.to_csv(out / "stability" / "task1_item_discrimination.csv", index=False)
    usable = item_disc.dropna(subset=["discrimination_rpb"])
    good = (usable["discrimination_rpb"].abs() > 0.3).sum()
    print(f"  item_discrimination: {len(usable)} computable, {good} with |r_pb| > 0.3")

    print("\n[dissociations] Cross-task correlations...")
    xtask = compute_cross_task_correlations(scores)
    xtask.to_csv(out / "dissociations" / "cross_task_spearman.csv", index=False)
    print(f"  cross_task_spearman: {len(xtask)} task pairs")

    print("\n[dissociations] Per-model profiles...")
    scores.to_csv(out / "dissociations" / "per_model_profile.csv")

    dissoc = compute_dissociation_highlights(scores)
    dissoc.to_csv(out / "dissociations" / "dissociation_highlights.csv", index=False)
    print(f"  dissociation_highlights: {len(dissoc)} models ranked by rank-range")

    print("\n[summary] Writing summary_findings.md...")
    write_summary_findings(
        out / "summary_findings.md",
        scores, dg, reward_sens, diff_sens, stab_summary, item_disc, xtask, dissoc,
    )
    print(f"\n✓ Analysis complete. Outputs in {out}")


if __name__ == "__main__":
    main()
