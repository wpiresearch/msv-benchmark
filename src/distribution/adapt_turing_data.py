#!/usr/bin/env python3
"""
adapt_turing_data.py
====================

Adapts the per-model `results-<dataset>-<date>/<model>/exp2a_delegate_trials.csv`
files produced by the Turing Phase 2 Delegate Game runs into the canonical
column format expected by the analysis scripts (`compute_bootstrap_ci.py`,
`compute_rank_divergence_ci.py`).

This is the Turing-cohort equivalent of `adapt_kaggle_data.py` for the
Kaggle cohort.

Inputs (from results-<dataset>-<date>/):
  <model_tag>/exp2a_delegate_trials.csv
      columns (Turing schema): question_id, choice, answer, confidence,
                               would_be_correct, ...

  Optional difficulty file:
    <difficulty_csv> with columns question_id, difficulty
    (e.g. gpqa_difficulty_scores.csv computed from the same cohort).
    If omitted, the `difficulty` column is filled with NaN; the bootstrap
    script tolerates this.

Outputs (to <output_dir>/):
  delegate_game/<model_tag>.csv    # canonical columns:
      question_id, answer, correct, confidence, delegated, difficulty

  where:
    - delegated = (choice == "DELEGATE") as 0/1
    - correct   = would_be_correct as 0/1 on ANSWER rows; 0 on DELEGATE rows
    - confidence = NaN on DELEGATE rows
    - answer    = NaN on DELEGATE rows

Usage:
    python adapt_turing_data.py \
        --turing_results_dir ~/msv_benchmark/kaggle_neurips/results/results-gpqa-2026-03-25 \
        --output_dir         ~/msv_benchmark/kaggle_neurips/results/reproduced/turing_analysis_input/

    # With a difficulty file:
    python adapt_turing_data.py \
        --turing_results_dir ~/msv_benchmark/kaggle_neurips/results/results-gpqa-2026-03-25 \
        --difficulty_csv     ~/msv_benchmark/kaggle_neurips/data/gpqa_difficulty_scores.csv \
        --output_dir         ~/msv_benchmark/kaggle_neurips/results/reproduced/turing_analysis_input/

Dependencies: pandas, numpy
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def adapt_one_model(
    model_dir: Path,
    out_dir: Path,
    difficulty_lookup: dict | None,
) -> int:
    """Adapt one Turing per-model directory to canonical Task-1 format.

    Returns the number of rows written, or 0 if no trial CSV is present
    (e.g. partial-run model directories that only contain
    exp1_binary_pairs_msv_log.json with no exp2a output).
    """
    src = model_dir / "exp2a_delegate_trials.csv"
    if not src.exists():
        return 0
    df = pd.read_csv(src)

    # Required Turing columns
    needed = {"question_id", "choice"}
    missing = needed - set(df.columns)
    if missing:
        print(f"  WARNING: {model_dir.name} missing columns: {missing}", file=sys.stderr)
        return 0

    # Build canonical columns
    delegated = (df["choice"].astype(str).str.strip().str.upper() == "DELEGATE").astype(int)

    # would_be_correct may be bool, int, or string depending on pandas version
    if "would_be_correct" in df.columns:
        wbc = df["would_be_correct"]
        if wbc.dtype == object:
            wbc = wbc.astype(str).str.lower().map({"true": 1, "false": 0}).fillna(0)
        correct_int = wbc.fillna(0).astype(int)
    else:
        # Older Turing schema may use a 'correct' column directly
        correct_int = df.get("correct", pd.Series([0] * len(df))).fillna(0).astype(int)
    correct_int = correct_int.where(delegated == 0, 0)  # force 0 on DELEGATE rows

    answer_col = df["answer"] if "answer" in df.columns else pd.Series([pd.NA] * len(df))
    confidence_col = (pd.to_numeric(df["confidence"], errors="coerce")
                      if "confidence" in df.columns
                      else pd.Series([np.nan] * len(df)))

    # Difficulty: optional join from external file
    if difficulty_lookup is not None:
        difficulty = df["question_id"].astype(str).map(difficulty_lookup)
    else:
        difficulty = pd.Series([np.nan] * len(df))

    out = pd.DataFrame({
        "question_id": df["question_id"],
        "answer":      answer_col,
        "correct":     correct_int,
        "confidence":  confidence_col,
        "delegated":   delegated,
        "difficulty":  difficulty,
    })
    # Null out answer/confidence on DELEGATE rows
    out.loc[out["delegated"] == 1, ["answer", "confidence"]] = pd.NA

    dst_dir = out_dir / "delegate_game"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{model_dir.name}.csv"
    out.to_csv(dst, index=False)
    return len(out)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--turing_results_dir", required=True, type=Path,
                    help="Path to results-<dataset>-<date>/ directory "
                         "containing per-model subdirectories")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Where to write adapted CSVs (will create "
                         "delegate_game/ subdirectory)")
    ap.add_argument("--difficulty_csv", type=Path, default=None,
                    help="Optional CSV with columns question_id,difficulty "
                         "to populate the difficulty column. If omitted, "
                         "difficulty is left as NaN.")
    args = ap.parse_args()

    if not args.turing_results_dir.exists():
        sys.exit(f"ERROR: {args.turing_results_dir} does not exist")

    # Load difficulty lookup if provided
    difficulty_lookup = None
    if args.difficulty_csv is not None:
        if not args.difficulty_csv.exists():
            sys.exit(f"ERROR: {args.difficulty_csv} does not exist")
        diff_df = pd.read_csv(args.difficulty_csv)
        if "question_id" not in diff_df.columns or "difficulty" not in diff_df.columns:
            sys.exit(f"ERROR: {args.difficulty_csv} must have columns "
                     f"question_id,difficulty (got: {list(diff_df.columns)})")
        difficulty_lookup = dict(zip(diff_df["question_id"].astype(str),
                                     diff_df["difficulty"]))
        print(f"Loaded difficulty for {len(difficulty_lookup)} questions "
              f"from {args.difficulty_csv}")

    model_dirs = sorted(d for d in args.turing_results_dir.iterdir() if d.is_dir())
    print(f"Found {len(model_dirs)} model directories in {args.turing_results_dir}")

    n_total = 0
    n_written = 0
    n_skipped = 0
    for md in model_dirs:
        n_rows = adapt_one_model(md, args.output_dir, difficulty_lookup)
        if n_rows > 0:
            print(f"  {md.name:<30} -> delegate_game/{md.name}.csv ({n_rows} rows)")
            n_written += 1
            n_total += n_rows
        else:
            print(f"  {md.name:<30} -> SKIPPED (no exp2a_delegate_trials.csv)")
            n_skipped += 1

    print(f"\nWrote {n_written} adapted CSVs ({n_total} total rows); "
          f"skipped {n_skipped} model directories")
    print(f"Output: {args.output_dir / 'delegate_game'}")
    if difficulty_lookup is None:
        print("\nNote: difficulty column is NaN (no --difficulty_csv provided). "
              "compute_bootstrap_ci.py tolerates this; "
              "compute_rank_divergence_ci.py does not require difficulty.")


if __name__ == "__main__":
    main()
