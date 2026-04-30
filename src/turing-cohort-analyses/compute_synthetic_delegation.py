#!/usr/bin/env python3
"""
================================================================================
MSV: Kaggle: NeurIPS E&D 2026 -- Synthetic Delegation Baseline (v2)
================================================================================
Project     : MSV Metacognition Benchmark
Paper       : "Beyond Confidence Calibration: Behavioral Metacognitive Control
               as a Distinct Evaluation Target for Large Language Models"
Track       : NeurIPS 2026 Evaluations & Datasets
Authors     : Ricky J. Sethi, Hefei Qiu, Mina Fahmy

Purpose
-------
Compare actual Delegate Game delegation decisions against a "synthetic
delegation" baseline derived from confidence thresholding.

This comparison is central to the paper's thesis: if actual behavioral
delegation diverges from what a confidence-threshold rule would predict,
then behavioral delegation captures information not present in the model's
stated confidence.

Revision Notes (v2)
-------------------
The v1 script assigned confidence=0 to delegated items, which creates a
methodological circularity: the synthetic rule trivially matches actual
delegation on those rows because both produce "delegate." This inflates
agreement and obscures the true distinction between behavioral routing
and confidence thresholding.

The v2 design requires forced-answer data (Phase 1) in which the model
answered EVERY question with a confidence rating and no delegation option.
Synthetic delegation is then computed from the forced-answer confidence:
"delegate iff forced-answer confidence < t." This is compared against the
model's actual delegation decisions from the Delegate Game (Phase 2).

When forced-answer data is unavailable, the script operates in a degraded
mode using answered-only trials from the Delegate Game, but explicitly
warns that the comparison is limited to the subset where the model chose
to answer, which is not the full item set.

Inputs
------
Delegate Game CSV (one per model):
    question_id, answer, correct, confidence, delegated, difficulty

Forced-answer CSV (one per model, same naming convention):
    question_id, answer, correct, confidence
    (No delegation; model answered all questions.)

Outputs
-------
- synthetic_delegation_summary.csv : per-model, per-threshold comparison

Usage
-----
    python compute_synthetic_delegation.py \
        --delegate_dir ./results/delegate_game/ \
        --forced_answer_dir ./results/forced_answer/ \
        --output_dir ./results/synthetic/

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
from sklearn.metrics import cohen_kappa_score, roc_auc_score


# ============================================================================
# Analysis
# ============================================================================

def compute_delegation_auc(correct, delegation_signal):
    """AUC-ROC of a delegation signal as predictor of incorrectness."""
    labels = (1 - correct).astype(int)
    if len(np.unique(labels)) < 2 or len(np.unique(delegation_signal)) < 2:
        return np.nan
    return roc_auc_score(labels, delegation_signal)


def analyze_model(dg_df, fa_df, model_name, thresholds=None):
    """
    Compare actual delegation against forced-answer confidence thresholding.

    Parameters
    ----------
    dg_df : pd.DataFrame
        Delegate Game results (Phase 2).
    fa_df : pd.DataFrame
        Forced-answer results (Phase 1). Must contain confidence for ALL
        questions.
    model_name : str
        Model identifier.
    thresholds : list of int
        Confidence thresholds to test.

    Returns
    -------
    list of dict
        One entry per threshold.
    """
    if thresholds is None:
        thresholds = [1, 2, 3, 4]

    # Merge on question_id to align Phase 1 and Phase 2 data
    merged = dg_df[["question_id", "correct", "delegated", "difficulty"]].merge(
        fa_df[["question_id", "confidence"]].rename(
            columns={"confidence": "fa_confidence"}
        ),
        on="question_id", how="inner"
    )

    if len(merged) == 0:
        print(f"  WARNING: No overlapping questions between Delegate Game "
              f"and forced-answer data for {model_name}.", file=sys.stderr)
        return []

    correct = merged["correct"].values.astype(int)
    actual_deleg = merged["delegated"].values.astype(int)
    fa_conf = merged["fa_confidence"].values.astype(int)

    # Actual delegation AUC (using Delegate Game correctness labels)
    actual_auc = compute_delegation_auc(correct, actual_deleg)

    results = []

    for t in thresholds:
        # Synthetic delegation: delegate iff forced-answer confidence < t
        synth_deleg = (fa_conf < t).astype(int)

        # Agreement
        agreement_rate = np.mean(actual_deleg == synth_deleg)

        # Cohen's kappa (chance-corrected agreement)
        if (len(np.unique(actual_deleg)) < 2 or
                len(np.unique(synth_deleg)) < 2):
            kappa = np.nan
        else:
            kappa = cohen_kappa_score(actual_deleg, synth_deleg)

        # Synthetic AUC
        synth_auc = compute_delegation_auc(correct, synth_deleg)

        # Disagreement analysis
        disagree_mask = actual_deleg != synth_deleg
        n_disagree = disagree_mask.sum()

        if n_disagree > 0:
            actual_better = 0
            synth_better = 0
            for i in np.where(disagree_mask)[0]:
                if actual_deleg[i] == 1 and synth_deleg[i] == 0:
                    # Actual delegated, synthetic did not
                    if correct[i] == 0:
                        actual_better += 1  # Correctly identified failure
                    else:
                        synth_better += 1   # Unnecessary delegation
                elif actual_deleg[i] == 0 and synth_deleg[i] == 1:
                    # Actual answered, synthetic would have delegated
                    if correct[i] == 1:
                        actual_better += 1  # Correctly kept
                    else:
                        synth_better += 1   # Should have delegated
            actual_win_rate = actual_better / n_disagree
        else:
            actual_win_rate = np.nan

        results.append({
            "model": model_name,
            "threshold": t,
            "n_items": len(merged),
            "agreement_rate": agreement_rate,
            "cohen_kappa": kappa,
            "actual_auc": actual_auc,
            "synthetic_auc": synth_auc,
            "auc_difference": (actual_auc - synth_auc)
                if not (np.isnan(actual_auc) or np.isnan(synth_auc))
                else np.nan,
            "n_disagree": n_disagree,
            "actual_win_rate_on_disagree": actual_win_rate,
            "actual_deleg_rate": np.mean(actual_deleg),
            "synthetic_deleg_rate": np.mean(synth_deleg),
        })

    return results


def analyze_model_degraded(dg_df, model_name, thresholds=None):
    """
    Degraded mode: no forced-answer data available.

    Uses only answered trials from the Delegate Game. The comparison is
    limited because delegated items are excluded, so we cannot test whether
    the model's delegation decision diverges from what confidence thresholding
    would have predicted for those items.

    This mode explicitly warns in output that the analysis is incomplete.
    """
    if thresholds is None:
        thresholds = [1, 2, 3, 4]

    answered = dg_df[dg_df["delegated"] == 0].copy()
    if len(answered) < 5:
        return []

    correct = answered["correct"].values.astype(int)
    conf_raw = answered["confidence"].values.astype(int)

    results = []
    for t in thresholds:
        # On answered items, "would this model have delegated under a
        # threshold rule?"
        synth_deleg = (conf_raw < t).astype(int)

        # But actual delegation on these items is always 0 (they answered),
        # so kappa and agreement are computed between "always 0" and
        # the synthetic rule. This is not very informative.

        results.append({
            "model": model_name,
            "threshold": t,
            "n_items": len(answered),
            "note": "DEGRADED: answered-only, no forced-answer data",
            "synthetic_deleg_rate_on_answered": np.mean(synth_deleg),
            "accuracy_if_synthetic_answered": (
                correct[synth_deleg == 0].mean()
                if (synth_deleg == 0).sum() > 0 else np.nan
            ),
        })

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare actual Delegate Game delegation against "
                    "confidence-threshold synthetic delegation."
    )
    parser.add_argument("--delegate_dir", type=str, required=True,
                        help="Directory with Delegate Game CSVs.")
    parser.add_argument("--forced_answer_dir", type=str, default=None,
                        help="Directory with forced-answer CSVs. Required "
                             "for a methodologically clean comparison.")
    parser.add_argument("--output_dir", type=str,
                        default="./results/synthetic/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dg_files = sorted(glob.glob(os.path.join(args.delegate_dir, "*.csv")))
    if not dg_files:
        print(f"ERROR: No CSVs in {args.delegate_dir}", file=sys.stderr)
        sys.exit(1)

    # Load forced-answer data
    fa_data = {}
    if args.forced_answer_dir:
        for fpath in glob.glob(os.path.join(args.forced_answer_dir, "*.csv")):
            mname = os.path.splitext(os.path.basename(fpath))[0]
            try:
                fa_data[mname] = pd.read_csv(fpath)
            except Exception as e:
                print(f"WARNING: {fpath}: {e}", file=sys.stderr)

    if not fa_data:
        print("=" * 70)
        print("WARNING: No forced-answer data provided.")
        print("The synthetic delegation comparison requires forced-answer")
        print("confidence data (Phase 1) where the model answered every")
        print("question without a delegation option. Without this data,")
        print("the analysis is limited to answered-only trials and the")
        print("comparison is methodologically incomplete.")
        print("=" * 70)
        print()

    all_results = []

    for fpath in dg_files:
        mname = os.path.splitext(os.path.basename(fpath))[0]
        try:
            dg_df = pd.read_csv(fpath)
        except Exception as e:
            print(f"WARNING: {fpath}: {e}", file=sys.stderr)
            continue

        fa_df = fa_data.get(mname, None)

        if fa_df is not None:
            print(f"Processing: {mname} (forced-answer available)")
            results = analyze_model(dg_df, fa_df, mname)
        else:
            print(f"Processing: {mname} (DEGRADED: no forced-answer)")
            results = analyze_model_degraded(dg_df, mname)

        all_results.extend(results)

        for r in results:
            if "cohen_kappa" in r:
                print(f"  t={r['threshold']}: kappa={r['cohen_kappa']:.3f}, "
                      f"AUC(actual)={r['actual_auc']:.3f}, "
                      f"AUC(synth)={r['synthetic_auc']:.3f}")
            else:
                print(f"  t={r['threshold']}: {r.get('note', '')}")
        print()

    if all_results:
        summary_df = pd.DataFrame(all_results)
        out_path = os.path.join(args.output_dir,
                                "synthetic_delegation_summary.csv")
        summary_df.to_csv(out_path, index=False)
        print(f"Summary saved to: {out_path}")


if __name__ == "__main__":
    main()
