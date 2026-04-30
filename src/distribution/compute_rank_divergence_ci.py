#!/usr/bin/env python3
"""
================================================================================
MSV: Bootstrap CIs on cross-model rank divergence (tau, rho, top-k overlap)
================================================================================

Purpose
-------
Compute bootstrap confidence intervals on the central rank-divergence claim
(Kendall tau, Spearman rho) between the declarative-ranking (by ECE) and
the behavioral-ranking (by Delegation AUC-ROC) across the subset of models
with both metrics computable. Also reports pairwise rank-reversal
magnitudes (|ECE_rank - DelegAUC_rank|) with bootstrap CIs.

Motivation
----------
The paper's primary estimand is tau = +0.20 on the 11 Kaggle models
with both ECE and own-error Delegation AUC computable. A point
estimate alone does not tell the reader whether tau is
distinguishable from some reference value (0 = no association,
1 = perfect agreement). Bootstrap CIs over the question-level resampling
distribution provide that uncertainty.

(An earlier revision of this paper reported tau = 0.018 as the
primary value; that estimate used delegation AUC against dataset
hardness rather than against own error. See the rank-divergence
audit in results/kaggle_cohort/rank_divergence_audit.md for details.)

Note that the bootstrap resampling here is over QUESTIONS (the unit of
independent sampling), not over models. Resampling over models would
confuse inter-model and inter-question variance.

Methodology
-----------
For each of B bootstrap iterations:
  1. Resample the 80 questions with replacement.
  2. Recompute ECE and Delegation AUC-ROC per model on the resampled
     question set.
  3. Re-filter models to those with both metrics computable (delegation
     rate > 0% and < 100%, and answered count >= min_answered).
  4. Rank models on ECE (ascending, 1=best) and on Delegation AUC-ROC
     (descending, 1=best).
  5. Compute Kendall tau and Spearman rho between the two ranking vectors.
  6. Also compute max pairwise |rank_diff| across surviving models.

Output is the bootstrap distribution plus summary statistics.

Usage
-----
  # Primary: own-error AUC (default)
  python compute_rank_divergence_ci.py \\
      --input_dir     ./analysis_input/delegate_game/ \\
      --n_boot        10000 \\
      --min_answered  20 \\
      --output_csv    ./results/rank_divergence_bootstrap.csv

  # Alternative behavioral target: global-hardness AUC
  python compute_rank_divergence_ci.py \\
      --input_dir     ./analysis_input/delegate_game/ \\
      --auc_target    hardness \\
      --n_boot        10000 \\
      --min_answered  20 \\
      --output_csv    ./results/rank_divergence_bootstrap_hardness.csv

The --auc_target flag selects between the paper's primary behavioral
metric (own-error AUC: does the model delegate items it would itself
get wrong?) and the alternative target reported in the appendix audit
(global-hardness AUC: does the model delegate items that are objectively
difficult relative to the fixed evaluation panel?). The two targets
answer different scientific questions and should not be treated as
interchangeable; see the appendix `app:rank_divergence_audit` and the
`compute_delegation_auc_vs_hardness` docstring for the conceptual
distinction.

For hardness AUC, the threshold is computed once across the union of
unique (question_id, difficulty) pairs in the input model CSVs and
attached as a stable hard_label column before bootstrapping. This
contrasts with the per-model median used in analyze_kaggle_cohort.py
(which produces the deleg_auc_vs_hardness_per_model_median column);
the global-median version computed here is the methodologically
preferred form and is what the paper appendix reports.

Inputs
------
Per-model CSVs with columns:
    question_id, answer, correct, confidence, delegated, difficulty

(matching the schema of compute_bootstrap_ci.py; use adapt_kaggle_data.py
to produce them from the extractor output. The 'difficulty' column is
required when --auc_target=hardness.)

Dependencies
------------
numpy, pandas, scipy, scikit-learn
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import roc_auc_score


CONFIDENCE_TO_PROB = {1: 0.25, 2: 0.50, 3: 0.75, 4: 1.00}
# This mapping matches the paper's Section 4.3 declarative-baselines
# definition: "Confidence ratings 1--4 are mapped linearly to
# probabilities {0.25, 0.50, 0.75, 1.00}." The same mapping is used in
# compute_bootstrap_ci.py to ensure ECE values are directly comparable
# across the two analysis scripts. (Earlier revisions of this script
# used {1: 0.0, 2: 1/3, 3: 2/3, 4: 1.0}, which the paper's Section 4.3
# does not endorse; the change to bin-midpoints aligns this script
# with the paper text and with compute_bootstrap_ci.py.)


def load_forced_answer_dir(forced_answer_dir):
    """Load forced-answer CSVs from a directory into a dict keyed by model name.

    Each CSV should have columns: question_id, correct, confidence, parse_failure.
    Returns {model_name: DataFrame} with parse_failure rows excluded.
    Skips qc_summary.csv and other non-model CSVs by checking column structure.
    """
    if forced_answer_dir is None:
        return {}
    fa_dir = Path(forced_answer_dir)
    if not fa_dir.exists():
        print(f"WARNING: forced_answer_dir {fa_dir} does not exist", file=sys.stderr)
        return {}
    fa_dfs = {}
    for fp in sorted(glob.glob(str(fa_dir / "*.csv"))):
        name = Path(fp).stem
        df = pd.read_csv(fp)
        # Skip non-model CSVs (qc_summary, model_name_map, etc.)
        if "question_id" not in df.columns or "confidence" not in df.columns:
            continue
        # Drop parse failures
        if "parse_failure" in df.columns:
            pf_mask = df["parse_failure"].astype(str).str.lower().isin(["true", "1"])
            df = df[~pf_mask].copy()
        fa_dfs[name] = df
    return fa_dfs


def compute_ece(correct: np.ndarray, confidence: np.ndarray, n_bins: int = 4) -> float:
    """ECE grouped by discrete confidence label.

    Per paper Section 4.3: one bin per discrete confidence level
    (k in {1, 2, 3, 4}), NOT equal-width binning on [0, 1]. The
    equal-width approach would silently merge confidence levels 3
    and 4 under the canonical mapping {0.25, 0.50, 0.75, 1.00}.
    The n_bins parameter is retained for backward compatibility but
    unused.

    `confidence` is expected to contain integer labels in {1, 2, 3, 4}.
    """
    del n_bins  # retained in signature for API compatibility; unused
    valid = ~np.isnan(confidence)
    if valid.sum() < 2:
        return np.nan
    c = correct[valid].astype(float)
    labels = confidence[valid].astype(int)
    n = len(c)
    ece = 0.0
    for k in (1, 2, 3, 4):
        mask = (labels == k)
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_k = CONFIDENCE_TO_PROB[k]
        acc_k = c[mask].mean()
        ece += (nb / n) * abs(acc_k - p_k)
    return ece


def compute_delegation_auc(correct: np.ndarray, delegated: np.ndarray) -> float:
    """AUC of delegation as classifier for own incorrectness.

    Behavioral target: own-error AUC. Asks whether the model delegates
    items it would itself get wrong. This is the paper's primary
    behavioral metric (Section 4.3) because it operationalizes the
    conceptual claim about behavioral self-monitoring.
    """
    labels = 1 - correct
    if len(np.unique(labels)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    try:
        return roc_auc_score(labels, delegated)
    except ValueError:
        return np.nan


def compute_delegation_auc_vs_hardness(
    delegated: np.ndarray, hard_label: np.ndarray
) -> float:
    """AUC of delegation as classifier for global item hardness.

    Behavioral target: global-hardness AUC. Asks whether the model
    delegates items that are objectively difficult, where "difficult"
    is defined relative to a fixed evaluation panel rather than the
    model's own self-selected subset. Reported in the paper as an
    alternative behavioral target in the rank-divergence audit
    (Appendix `app:rank_divergence_audit`), NOT as a sensitivity check
    of the own-error result. Own-error AUC and global-hardness AUC
    answer different scientific questions (own correctness prediction
    vs item-difficulty prediction).

    `hard_label` is expected to be a per-trial 0/1 array where 1 means
    "this question is at or above the global panel difficulty median"
    and 0 means below. The hard_label MUST be precomputed from the
    fixed evaluation panel and attached to each model's data BEFORE
    bootstrapping; computing it per-model or per-bootstrap-iteration
    introduces a self-selection confound.
    """
    if len(np.unique(hard_label)) < 2 or len(np.unique(delegated)) < 2:
        return np.nan
    try:
        return roc_auc_score(hard_label, delegated)
    except ValueError:
        return np.nan


def attach_global_hardness_label(model_dfs: dict) -> tuple:
    """Compute the global panel difficulty median across all models, then
    attach a `hard_label` column to each df indicating whether each
    trial's question is at or above that median.

    Returns (model_dfs_with_label, threshold_value, panel_n_unique_qs)
    where:
      - model_dfs_with_label is a new dict with hard_label attached
      - threshold_value is the median used (a scalar)
      - panel_n_unique_qs is the size of the unique-question panel that
        the median was computed over

    The median is computed once over unique (question_id, difficulty)
    pairs across the union of all models' data. This makes hardness a
    property of the question (and of the fixed evaluation panel),
    invariant to which model saw which subset and invariant to bootstrap
    resampling of question_ids.

    Raises ValueError if no model has a 'difficulty' column or if the
    union panel has fewer than 2 unique difficulty values.
    """
    panel_pairs = []
    for m, df in model_dfs.items():
        if "difficulty" not in df.columns:
            raise ValueError(
                f"model {m!r}: input df is missing required 'difficulty' "
                f"column needed for global-hardness AUC. The bootstrap "
                f"input schema must include 'difficulty' (see header "
                f"docstring); use adapt_kaggle_data.py / adapt_turing_data.py "
                f"with --include-difficulty to produce it."
            )
        sub = df[["question_id", "difficulty"]].dropna().drop_duplicates(
            subset=["question_id"]
        )
        panel_pairs.append(sub)
    if not panel_pairs:
        raise ValueError("model_dfs is empty; no panel to compute hardness over")
    panel_df = pd.concat(panel_pairs, ignore_index=True).drop_duplicates(
        subset=["question_id"]
    )
    if len(panel_df) < 2:
        raise ValueError(
            f"global panel has only {len(panel_df)} unique question(s); "
            f"cannot compute a meaningful hardness median"
        )
    threshold = float(panel_df["difficulty"].median())
    panel_n = len(panel_df)
    # Attach hard_label to each model's df
    qid_to_hard = dict(
        zip(panel_df["question_id"], (panel_df["difficulty"] >= threshold).astype(int))
    )
    out = {}
    for m, df in model_dfs.items():
        df_copy = df.copy()
        df_copy["hard_label"] = df_copy["question_id"].map(qid_to_hard).astype(
            "Int64"
        )
        out[m] = df_copy
    return out, threshold, panel_n


def metrics_for_model(df: pd.DataFrame, min_answered: int,
                       fa_df: pd.DataFrame = None,
                       auc_target: str = "own_error") -> dict:
    """Returns {ece, deleg_auc, n_answered, n_decl, deleg_rate,
    declarative_source} or NaN for any metric that is not computable
    under the filter criteria.

    If fa_df is provided (forced-answer DataFrame for this model), ECE
    is computed from forced-answer data on the intersection of
    question_ids between the behavioral df and fa_df. Otherwise ECE
    uses the answered-only subset of df (the legacy answered-conditional
    estimand). The same per-model swap pattern is implemented in
    compute_bootstrap_ci.py.

    auc_target controls which behavioral target the Delegation AUC is
    computed against:
      "own_error" (default): AUC labels are 1 - correct (per-trial
        per-model). Measures whether delegation tracks the model's own
        actual incorrectness. This is the paper's primary metric.
      "hardness": AUC labels are df["hard_label"] (per-trial). Requires
        df to have a hard_label column precomputed from the global
        evaluation panel difficulty median. Measures whether delegation
        tracks objective item hardness. Reported in the appendix as an
        alternative behavioral target.

    For auc_target="hardness", the hard_label column MUST be attached
    to df before this function is called; computing it on a per-model
    basis (using model-internal medians) is methodologically different
    and is not supported by this function.
    """
    if auc_target not in ("own_error", "hardness"):
        raise ValueError(
            f"auc_target must be 'own_error' or 'hardness', got {auc_target!r}"
        )
    if auc_target == "hardness" and "hard_label" not in df.columns:
        raise ValueError(
            "auc_target='hardness' requires df to have a 'hard_label' column "
            "(precomputed from the global panel difficulty median; see "
            "attach_global_hardness_label in this module)."
        )
    correct = df["correct"].values.astype(float)
    delegated = df["delegated"].values.astype(int)
    # Answered trials (always reported, even if fa_df is used for ECE)
    answered_mask = delegated == 0
    n_answered = int(answered_mask.sum())
    deleg_rate = float(delegated.mean())
    # ECE: from forced-answer if provided, else from answered-only subset
    if fa_df is not None and len(fa_df) > 0:
        # Inner-merge on question_id; behavioral df only contributes the
        # keys. This avoids _x/_y collisions and uses fa_df as authoritative
        # for both correct and confidence (forced-answer estimand).
        merged = df[["question_id"]].merge(
            fa_df[["question_id", "correct", "confidence"]],
            on="question_id", how="inner",
        )
        if len(merged) >= min_answered:
            decl_correct = merged["correct"].values.astype(float)
            decl_conf = merged["confidence"].values.astype(float)
            ece = compute_ece(decl_correct, decl_conf)
            declarative_source = "forced_answer"
            n_decl = len(merged)
        else:
            ece = np.nan
            declarative_source = "forced_answer_below_min"
            n_decl = len(merged)
    else:
        if n_answered < min_answered:
            ece = np.nan
            declarative_source = "answered_only_below_min"
        else:
            conf = df.loc[answered_mask, "confidence"].values.astype(float)
            ece = compute_ece(correct[answered_mask], conf)
            declarative_source = "answered_only"
        n_decl = n_answered
    # Delegation AUC: requires mixed behavior; always from behavioral df.
    # The behavioral target is chosen by auc_target ("own_error" default,
    # "hardness" for the alternative-target audit reported in the appendix).
    if deleg_rate == 0.0 or deleg_rate == 1.0:
        deleg_auc = np.nan
    elif auc_target == "own_error":
        deleg_auc = compute_delegation_auc(correct, delegated)
    elif auc_target == "hardness":
        hard_label = df["hard_label"].values.astype(int)
        deleg_auc = compute_delegation_auc_vs_hardness(delegated, hard_label)
    else:
        raise ValueError(f"unreachable: auc_target={auc_target!r}")
    return {
        "ece": ece,
        "deleg_auc": deleg_auc,
        "n_answered": n_answered,
        "n_decl": n_decl,
        "deleg_rate": deleg_rate,
        "declarative_source": declarative_source,
    }


def rank_divergence_one_sample(model_dfs: dict, min_answered: int,
                                fa_dfs: dict = None,
                                auc_target: str = "own_error"):
    """Compute tau, rho, and max |rank_diff| on one (possibly bootstrap-
    resampled) view of the data.

    Returns (tau, rho, max_rank_diff, n_computable) with NaN where fewer
    than 3 models survive the filter. If fa_dfs is provided, ECE comes
    from forced-answer data per model where available. auc_target
    selects the behavioral target ("own_error" or "hardness"); see
    metrics_for_model for the semantics.
    """
    fa_dfs = fa_dfs or {}
    metrics = {
        m: metrics_for_model(df, min_answered, fa_df=fa_dfs.get(m),
                             auc_target=auc_target)
        for m, df in model_dfs.items()
    }
    computable = {
        m: v for m, v in metrics.items()
        if not (np.isnan(v["ece"]) or np.isnan(v["deleg_auc"]))
    }
    n = len(computable)
    if n < 3:
        return np.nan, np.nan, np.nan, n
    models = sorted(computable.keys())
    ece_vec   = np.array([computable[m]["ece"]       for m in models])
    auc_vec   = np.array([computable[m]["deleg_auc"] for m in models])
    ece_rank  = pd.Series(ece_vec).rank(ascending=True,  method="min").values
    auc_rank  = pd.Series(auc_vec).rank(ascending=False, method="min").values
    tau, _ = kendalltau(ece_rank, auc_rank)
    rho, _ = spearmanr(ece_rank, auc_rank)
    max_diff = float(np.max(np.abs(ece_rank - auc_rank)))
    return float(tau), float(rho), max_diff, n


def bootstrap_rank_divergence(model_dfs: dict, n_boot: int,
                              min_answered: int, seed: int = 42,
                              fa_dfs: dict = None,
                              auc_target: str = "own_error") -> dict:
    """Run the bootstrap and return summary statistics.

    Bootstrap resampling: for each iteration we resample each model's
    own question panel with replacement. This preserves the structural
    feature that different models have different coverage of the 80
    items (due to completion failures on the Kaggle platform), while
    still capturing question-level sampling uncertainty.

    auc_target selects the behavioral target ("own_error" or
    "hardness"); see metrics_for_model for the semantics. When
    auc_target="hardness", each df in model_dfs must have a
    pre-computed hard_label column; this label is carried through
    bootstrap resampling because it is a question-level (not
    bootstrap-iteration-level) property.
    """
    fa_dfs = fa_dfs or {}
    # First, point estimate on the full data
    tau_pt, rho_pt, maxdiff_pt, n_pt = rank_divergence_one_sample(
        model_dfs, min_answered, fa_dfs=fa_dfs, auc_target=auc_target,
    )
    print(f"Point estimate: tau={tau_pt:.3f}, rho={rho_pt:.3f}, "
          f"max|rank diff|={maxdiff_pt:.0f}, n_computable={n_pt}")

    # Record per-model question panels
    model_panels = {m: df["question_id"].unique().tolist()
                    for m, df in model_dfs.items()}
    model_indexed = {m: df.set_index("question_id") for m, df in model_dfs.items()}

    rng = np.random.RandomState(seed)
    tau_b, rho_b, maxdiff_b, n_b = [], [], [], []
    for b in range(n_boot):
        # For each model, resample its own panel with replacement
        resampled_dfs = {}
        for m, panel in model_panels.items():
            if len(panel) == 0:
                continue
            resampled_qs = rng.choice(panel, size=len(panel), replace=True)
            sub = model_indexed[m].reindex(resampled_qs).reset_index().dropna(
                subset=["correct", "delegated"]
            )
            if len(sub) > 0:
                resampled_dfs[m] = sub
        if len(resampled_dfs) < 3:
            continue
        t, r, d, n = rank_divergence_one_sample(
            resampled_dfs, min_answered, fa_dfs=fa_dfs,
            auc_target=auc_target,
        )
        if not (np.isnan(t) or np.isnan(r)):
            tau_b.append(t); rho_b.append(r); maxdiff_b.append(d); n_b.append(n)
        if (b + 1) % max(n_boot // 10, 1) == 0:
            print(f"  {b+1}/{n_boot} bootstrap samples...", file=sys.stderr)

    tau_b, rho_b = np.asarray(tau_b), np.asarray(rho_b)
    maxdiff_b, n_b = np.asarray(maxdiff_b), np.asarray(n_b)

    def _summ(arr):
        if len(arr) == 0:
            return (np.nan,) * 4
        return (float(np.mean(arr)), float(np.std(arr)),
                float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    # Record declarative_source distribution for the point estimate
    pt_metrics = {
        m: metrics_for_model(df, min_answered, fa_df=fa_dfs.get(m),
                             auc_target=auc_target)
        for m, df in model_dfs.items()
    }
    src_counts = {}
    for m, v in pt_metrics.items():
        s = v.get("declarative_source", "unknown")
        src_counts[s] = src_counts.get(s, 0) + 1

    summary = {
        "auc_target":       auc_target,
        "tau_point":        tau_pt,
        "tau_mean":         _summ(tau_b)[0],
        "tau_se":           _summ(tau_b)[1],
        "tau_ci_lo":        _summ(tau_b)[2],
        "tau_ci_hi":        _summ(tau_b)[3],
        "rho_point":        rho_pt,
        "rho_mean":         _summ(rho_b)[0],
        "rho_se":           _summ(rho_b)[1],
        "rho_ci_lo":        _summ(rho_b)[2],
        "rho_ci_hi":        _summ(rho_b)[3],
        "maxdiff_point":    maxdiff_pt,
        "maxdiff_mean":     _summ(maxdiff_b)[0],
        "maxdiff_ci_lo":    _summ(maxdiff_b)[2],
        "maxdiff_ci_hi":    _summ(maxdiff_b)[3],
        "n_computable_point":  n_pt,
        "n_computable_median": float(np.median(n_b)) if len(n_b) else np.nan,
        "n_boot_valid":        len(tau_b),
        "n_boot_requested":    n_boot,
        "min_answered":        min_answered,
        "declarative_source_distribution": "|".join(
            f"{k}={v}" for k, v in sorted(src_counts.items())
        ),
    }
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_dir", type=Path, required=True,
                    help="Per-model CSVs matching adapt_kaggle_data.py output")
    ap.add_argument("--forced_answer_dir", type=Path, default=None,
                    help="Optional directory of forced-answer CSVs. When "
                         "provided, ECE is computed from forced-answer data "
                         "(full-panel estimand) instead of from the answered-"
                         "only subset (answered-conditional estimand) for any "
                         "model whose forced-answer CSV is present.")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--min_answered", type=int, default=5,
                    help="Minimum answered trials for ECE to be computed "
                         "(default: 5, matches paper's primary estimand "
                         "n=11 on the 23-model Kaggle cohort; use 20 for "
                         "stricter subset n=10 sensitivity analysis)")
    ap.add_argument("--auc_target", choices=("own_error", "hardness"),
                    default="own_error",
                    help="Behavioral target for Delegation AUC. "
                         "'own_error' (default) is the paper's primary "
                         "metric (delegation as classifier of own "
                         "incorrectness). 'hardness' is the alternative "
                         "behavioral target reported in the appendix "
                         "audit, computed against the global panel "
                         "difficulty median (NOT per-model). When "
                         "auc_target='hardness', input CSVs must include "
                         "a 'difficulty' column.")
    ap.add_argument("--output_csv", type=Path,
                    default=Path("./rank_divergence_bootstrap.csv"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv_files = sorted(glob.glob(str(args.input_dir / "*.csv")))
    if not csv_files:
        print(f"No CSVs in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    model_dfs = {}
    for fp in csv_files:
        name = Path(fp).stem
        df = pd.read_csv(fp)
        need_cols = {"question_id", "correct", "confidence", "delegated"}
        if args.auc_target == "hardness":
            need_cols.add("difficulty")
        if not need_cols.issubset(df.columns):
            print(f"  skip {name}: missing columns {need_cols - set(df.columns)}",
                  file=sys.stderr)
            continue
        model_dfs[name] = df

    # If hardness AUC, compute global panel median once and attach hard_label
    # to each model's df. The label is a question-level (not bootstrap-
    # iteration-level) property; computing it once before bootstrap ensures
    # the target labels do not shift across bootstrap replicates.
    hardness_threshold = None
    hardness_panel_n = None
    if args.auc_target == "hardness":
        try:
            model_dfs, hardness_threshold, hardness_panel_n = (
                attach_global_hardness_label(model_dfs)
            )
            print(f"  Global panel hardness threshold: "
                  f"{hardness_threshold:.4f} (median of {hardness_panel_n} "
                  f"unique question difficulty values across all input models)")
        except ValueError as e:
            print(f"ERROR computing global hardness label: {e}", file=sys.stderr)
            sys.exit(1)

    fa_dfs = load_forced_answer_dir(args.forced_answer_dir)
    n_fa = len(fa_dfs)
    print(f"Loaded {len(model_dfs)} models, {n_fa} forced-answer CSVs. "
          f"Running bootstrap ({args.n_boot} "
          f"iterations, min_answered={args.min_answered}, "
          f"auc_target={args.auc_target})...")
    if n_fa > 0:
        with_fa = sorted(set(fa_dfs.keys()) & set(model_dfs.keys()))
        without_fa = sorted(set(model_dfs.keys()) - set(fa_dfs.keys()))
        print(f"  Forced-answer ECE for: {', '.join(with_fa) if with_fa else '(none)'}")
        if without_fa:
            print(f"  Answered-only ECE for: {', '.join(without_fa)}")

    summary = bootstrap_rank_divergence(
        model_dfs, args.n_boot, args.min_answered, seed=args.seed,
        fa_dfs=fa_dfs, auc_target=args.auc_target,
    )

    # Add hardness provenance fields (only meaningful when auc_target=hardness,
    # but we record them in every output CSV for self-documentation)
    summary["hardness_threshold"] = (
        hardness_threshold if hardness_threshold is not None else ""
    )
    summary["hardness_panel_n"] = (
        hardness_panel_n if hardness_panel_n is not None else ""
    )

    # Write CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(args.output_csv, index=False)
    print(f"\nWrote: {args.output_csv}")

    # Human-readable print
    print("\nSummary:")
    print(f"  AUC target: {summary['auc_target']}")
    print(f"  Kendall tau:  {summary['tau_point']:.3f}  "
          f"[95% CI: {summary['tau_ci_lo']:.3f}, {summary['tau_ci_hi']:.3f}]  "
          f"(SE: {summary['tau_se']:.3f})")
    print(f"  Spearman rho: {summary['rho_point']:.3f}  "
          f"[95% CI: {summary['rho_ci_lo']:.3f}, {summary['rho_ci_hi']:.3f}]  "
          f"(SE: {summary['rho_se']:.3f})")
    print(f"  Max |rank diff|: {summary['maxdiff_point']:.0f}  "
          f"[95% CI: {summary['maxdiff_ci_lo']:.0f}, "
          f"{summary['maxdiff_ci_hi']:.0f}]")
    print(f"  n_computable: point {summary['n_computable_point']}, "
          f"bootstrap median {summary['n_computable_median']:.0f}")
    print(f"  Valid bootstrap samples: {summary['n_boot_valid']} / "
          f"{summary['n_boot_requested']}")


if __name__ == "__main__":
    main()
