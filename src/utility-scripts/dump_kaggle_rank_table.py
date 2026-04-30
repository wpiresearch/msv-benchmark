#!/usr/bin/env python3
"""
dump_kaggle_rank_table.py
==========================

Build the per-model rank-divergence audit table for the Kaggle cohort.

Reads the Kaggle delegate-game metrics CSV produced by analyze_kaggle_cohort.py
and emits a per-model table with ECE rank, Delegation AUC rank, and the
absolute rank difference. This table is the auditable per-model breakdown
underneath the aggregate τ reported in Section 5.2 of the paper.

The output table is keyed by model and includes:
    - n_trials, answer_rate, n_answered (so small-sample caveats are visible)
    - ece_4bin (computed under canonical mapping + label-grouped binning)
    - deleg_auc_vs_own_err (the conceptually-primary AUC target)
    - ece_rank, auc_rank, delta_rank

Filtering: only models with both metrics non-NaN are included
(12 of 23 Kaggle models have delegation rate 0% or 100% and so have
undefined Delegation AUC; these are excluded).

Usage:
    python dump_kaggle_rank_table.py \\
        --input-csv  results/kaggle_cohort/comparative/delegate_game_metrics.csv \\
        --output-csv results/reproduced/rank_table_canonical_labelgrouped.csv

Optional flags:
    --print-tau:  also print the point-estimate Kendall τ and Spearman ρ
                  computed directly from the table; should match the
                  bootstrap point estimate from compute_rank_divergence_ci.py.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to delegate_game_metrics.csv (Kaggle cohort, "
             "from analyze_kaggle_cohort.py)",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the per-model rank table",
    )
    ap.add_argument(
        "--print-tau",
        action="store_true",
        help="Also print the point Kendall τ and Spearman ρ computed from "
             "the table (sanity check: should match bootstrap point estimate)",
    )
    args = ap.parse_args()

    if not args.input_csv.exists():
        print(f"ERROR: input not found: {args.input_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} rows from {args.input_csv}")

    required_cols = {"model", "n_trials", "answer_rate", "ece_4bin",
                     "deleg_auc_vs_own_err"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: input CSV missing required columns: {missing}",
              file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        return 1

    # Add n_answered (rounded from n_trials * answer_rate)
    df["n_answered"] = (df["n_trials"] * df["answer_rate"]).round().astype(int)

    # Drop models with either metric missing (always-delegators / never-delegators
    # have undefined Delegation AUC)
    computable = df.dropna(subset=["ece_4bin", "deleg_auc_vs_own_err"]).copy()
    n_excluded = len(df) - len(computable)
    if n_excluded > 0:
        excluded = df.loc[
            df["ece_4bin"].isna() | df["deleg_auc_vs_own_err"].isna(), "model"
        ].tolist()
        print(
            f"  Excluded {n_excluded} model(s) with NaN ECE or Delegation AUC "
            f"(degenerate delegation policy or insufficient data): {excluded}"
        )

    # Compute ranks
    computable = computable.reset_index(drop=True)
    computable["ece_rank"] = computable["ece_4bin"].rank(
        ascending=True, method="min"
    ).astype(int)
    computable["auc_rank"] = computable["deleg_auc_vs_own_err"].rank(
        ascending=False, method="min"
    ).astype(int)
    computable["delta_rank"] = (
        computable["ece_rank"] - computable["auc_rank"]
    ).abs()

    # Order columns
    out_cols = [
        "model", "n_trials", "answer_rate", "n_answered",
        "ece_4bin", "deleg_auc_vs_own_err",
        "ece_rank", "auc_rank", "delta_rank",
    ]
    out = computable[out_cols].sort_values("delta_rank", ascending=False)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(out)} rows to {args.output_csv}")
    print()
    print("=== Per-model rank table (sorted by |Δrank|) ===")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()

    print(f"max |Δrank|: {int(computable['delta_rank'].max())}")
    print(f"top-3 Δrank cases:")
    for _, r in out.head(3).iterrows():
        print(
            f"  {r['model']}: ECE rank {r['ece_rank']}, "
            f"AUC rank {r['auc_rank']}, Δ={r['delta_rank']}"
        )

    if args.print_tau:
        tau, _ = kendalltau(computable["ece_rank"], computable["auc_rank"])
        rho, _ = spearmanr(computable["ece_rank"], computable["auc_rank"])
        print()
        print(f"=== Point estimates (n={len(computable)}) ===")
        print(f"  Kendall τ:   {tau:.4f}")
        print(f"  Spearman ρ:  {rho:.4f}")
        print(
            "  (These should match the bootstrap point estimates from "
            "compute_rank_divergence_ci.py exactly.)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
