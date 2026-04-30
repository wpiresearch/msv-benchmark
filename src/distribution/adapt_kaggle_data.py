#!/usr/bin/env python3
"""
adapt_kaggle_data.py
====================

Adapts the `kaggle_extracted/` output produced by `extract_kaggle_outputs.py`
into the column format expected by the existing analysis scripts
(`compute_comparative_baselines.py`, `compute_sensitivity_analysis.py`,
`compute_ranking_stability.py`, `compute_bootstrap_ci.py`).

This is the Kaggle-cohort equivalent of `adapt_turing_data.py` in the
reproducibility bundle for the Turing cohort.

Inputs (from kaggle_extracted/):
  per_model/<model>/t01_delegate_game.csv    # columns: question_id, choice, answer, confidence, correct, difficulty, score, raw_response
  per_model/<model>/t03_second_chance.csv    # columns: question_id, initial_answer, final_answer, revised, initial_correct, final_correct, score, raw_response_initial, raw_response_revision
  ...

Outputs (to analysis_input/):
  delegate_game/<model>.csv    # canonical columns for Task 1 analyses:
                               #   question_id, answer, correct, confidence, delegated, difficulty

The `delegated` column is derived as `choice == "DELEGATE"`. When a row is
delegated, the `answer`/`correct`/`confidence` columns may be null, and
downstream analyses (especially the comparative baselines) handle the
answered-only subset appropriately.

Usage:
    python adapt_kaggle_data.py \
        --extracted_dir ./kaggle_extracted \
        --output_dir   ./analysis_input

Dependencies: pandas
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def adapt_task1(extracted_model_dir: Path, out_dir: Path, model: str) -> int:
    """Adapt t01_delegate_game.csv to the canonical Task-1 format.

    Canonical format (matches adapt_turing_data.py):
        question_id, answer, correct, confidence, delegated, difficulty
    where `correct` is a 0/1 int for answered rows and 0 for delegated rows
    (consistent with the Turing adapter's convention of `fillna(0).astype(int)`).
    `confidence` is NaN for delegated rows.
    """
    src = extracted_model_dir / "t01_delegate_game.csv"
    if not src.exists():
        return 0
    df = pd.read_csv(src)
    # Kaggle columns: question_id, choice, answer, confidence, correct, difficulty, score, raw_response
    # Note: on Kaggle side, `correct` is the gold-standard letter (A/B/C/D).
    # The analysis scripts want `correct` as a 0/1 indicator.
    # Answered rows: 1 if answer == correct_letter else 0. Delegated rows: 0.

    delegated = (df["choice"].astype(str).str.strip().str.upper() == "DELEGATE").astype(int)
    # Letter comparison is only meaningful when not delegated
    letters_match = (df["answer"].astype(str).str.strip() == df["correct"].astype(str).str.strip())
    correct_int = letters_match.astype(int)
    correct_int = correct_int.where(delegated == 0, 0)  # force 0 on delegated rows

    out = pd.DataFrame({
        "question_id": df["question_id"],
        "answer":      df["answer"],
        "correct":     correct_int,
        "confidence":  pd.to_numeric(df["confidence"], errors="coerce"),
        "delegated":   delegated,
        "difficulty":  df["difficulty"],
    })
    # Null out answer/confidence on delegated rows (convention matches Turing adapter)
    out.loc[out["delegated"] == 1, ["answer", "confidence"]] = pd.NA

    dst_dir = out_dir / "delegate_game"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{model}.csv"
    out.to_csv(dst, index=False)
    return len(out)

    dst_dir = out_dir / "delegate_game"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{model}.csv"
    out.to_csv(dst, index=False)
    return len(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--extracted_dir", required=True, type=Path,
                    help="Output directory from extract_kaggle_outputs.py (expects per_model/ subdir)")
    ap.add_argument("--output_dir", required=True, type=Path,
                    help="Where to write adapted CSVs (will contain delegate_game/ subdir)")
    args = ap.parse_args()

    per_model_dir = args.extracted_dir / "per_model"
    if not per_model_dir.exists():
        sys.exit(f"ERROR: {per_model_dir} does not exist")

    models = sorted(d.name for d in per_model_dir.iterdir() if d.is_dir())
    print(f"Found {len(models)} model directories in {per_model_dir}")

    n_total = 0
    for model in models:
        n_rows = adapt_task1(per_model_dir / model, args.output_dir, model)
        print(f"  {model:<40} -> delegate_game/{model}.csv ({n_rows} rows)")
        n_total += n_rows

    print(f"\nWrote Task 1 adapted CSVs for {len(models)} models ({n_total} total rows)")
    print(f"Output: {args.output_dir / 'delegate_game'}")


if __name__ == "__main__":
    main()
