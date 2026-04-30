#!/usr/bin/env python3
"""
build_turing_rank_reversal_input.py
====================================

Build a 6-row CSV (model, ece, deleg_auc) for the institutional cohort
to feed into generate_rank_reversal_figure.py.

Reads the per-model bootstrap CI summary produced by step 4b
(compute_bootstrap_ci.py with --forced_answer_dir) and emits a
3-column CSV containing only the mixed-delegators (the 3 always-
delegating institutional models have undefined Delegation AUC and
are correctly NaN-filtered).

The ECE value is the forced-answer Phase 1 estimate (declarative_source
= 'forced_answer' in the input CSV), per the paper's institutional
forced-answer protocol.

Usage:
    python build_turing_rank_reversal_input.py \\
        --bootstrap-csv results/reproduced/bootstrap_institutional_with_fa/bootstrap_ci_summary.csv \\
        --output-csv    results/reproduced/turing_rank_reversal_input.csv

Input columns required: model, ece_point, deleg_auc_point, declarative_source
Output columns: model, ece, deleg_auc

Exit code:
    0 on success
    1 on input-file error or missing required columns
    2 if the output would have fewer than 3 mixed-delegators (insufficient
      for a meaningful rank-reversal figure)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--bootstrap-csv",
        type=Path,
        required=True,
        help="Path to step 4b bootstrap_ci_summary.csv "
             "(institutional cohort with forced-answer)",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path to write the 3-column CSV "
             "(model, ece, deleg_auc) for the figure script",
    )
    ap.add_argument(
        "--require-forced-answer",
        action="store_true",
        default=True,
        help="Require declarative_source=forced_answer for every output row "
             "(default: True; pass --no-require-forced-answer to disable)",
    )
    ap.add_argument(
        "--no-require-forced-answer",
        dest="require_forced_answer",
        action="store_false",
    )
    args = ap.parse_args()

    if not args.bootstrap_csv.exists():
        print(f"ERROR: input file not found: {args.bootstrap_csv}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.bootstrap_csv)

    required_cols = {"model", "ece_point", "deleg_auc_point", "declarative_source"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"ERROR: input CSV missing required columns: {missing}", file=sys.stderr)
        print(f"Found columns: {list(df.columns)}", file=sys.stderr)
        return 1

    print(f"Loaded {len(df)} rows from {args.bootstrap_csv}")

    if args.require_forced_answer:
        non_fa = (df["declarative_source"] != "forced_answer").sum()
        if non_fa > 0:
            non_fa_models = df.loc[
                df["declarative_source"] != "forced_answer", "model"
            ].tolist()
            print(
                f"ERROR: --require-forced-answer is set but {non_fa} rows do not "
                f"have declarative_source='forced_answer'.\n"
                f"  Non-forced-answer rows: {non_fa_models}",
                file=sys.stderr,
            )
            print(
                f"  Did you pass the wrong CSV (step 2 instead of step 4b)?",
                file=sys.stderr,
            )
            return 1
        print(f"  All {len(df)} rows have declarative_source=forced_answer (OK)")

    # Drop models with undefined Delegation AUC (always-delegators / never-delegators)
    mixed = df.dropna(subset=["deleg_auc_point"]).copy()
    n_excluded = len(df) - len(mixed)
    if n_excluded > 0:
        excluded_models = df.loc[df["deleg_auc_point"].isna(), "model"].tolist()
        print(
            f"  Excluded {n_excluded} model(s) with NaN Delegation AUC "
            f"(always-delegate or never-delegate): {excluded_models}"
        )

    if len(mixed) < 3:
        print(
            f"ERROR: only {len(mixed)} mixed-delegator(s) remain; "
            f"a rank-reversal figure requires at least 3 models.",
            file=sys.stderr,
        )
        return 2

    out = mixed[["model", "ece_point", "deleg_auc_point"]].rename(
        columns={"ece_point": "ece", "deleg_auc_point": "deleg_auc"}
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"\nWrote {len(out)} rows to {args.output_csv}")
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
