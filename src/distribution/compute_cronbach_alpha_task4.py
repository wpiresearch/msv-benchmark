#!/usr/bin/env python3
"""
compute_cronbach_alpha_task4.py
================================

Computes Cronbach's alpha for Task 4 (Confidence Entropy) on the Kaggle
cohort, with two balance conventions and per-domain stratification.

Rationale
---------
Section 6.3 limitation 10 notes that three concurrent benchmarks
(Cacioli 2026, Singh 2026, Parikh 2026) report per-task internal consistency
reliability via Cronbach's alpha. This script produces those numbers for
Task 4 (Confidence Entropy), which is the task with the most item-level
structure: each model produces a probability distribution over A/B/C/D per
question, and the normalized entropy of that distribution is the per-item
score.

Task 4 is chosen (over e.g., Task 1) because the per-trial score is
continuous and sensitive to calibration quality. Task 1 rewards use discrete
jumps (correct/incorrect, delegate hard/medium/easy) that inflate item-level
variance in ways that are hard to interpret as "metacognitive reliability."

Two balance conventions
-----------------------
Cronbach's alpha is defined on a complete item x respondent matrix. Partial
completions on the Kaggle hosted platform (budget/quota failures, API
errors) mean we must choose how to handle missing cells.

  (A) "Strict balanced panel": keep only items completed by all 23 models.
      Small k (the number of items completed by every model), well-defined
      alpha. This is the most conservative variant and is directly comparable
      to textbook alpha definitions.

  (B) "Inclusive panel at threshold t": keep items completed by at least
      t-of-23 models and drop the remaining models from that item. Produces
      larger k but the reported alpha is the standard formula applied to the
      subset of models that completed that item set. We report t=23 (strict),
      t=20, and t=15 so reviewers can see how the point estimate changes with
      inclusion threshold.

Per-domain stratification
-------------------------
Task 4 does not carry the GPQA Diamond category column directly; we join it
from Task 2 (declared probe), which carries the same question_id and the
category. Alpha is reported per category as well as pooled.

Usage
-----
    python compute_cronbach_alpha_task4.py \\
        --task4-csv ./kaggle_extracted/per_task/t04_confidence_entropy.csv \\
        --task2-csv ./kaggle_extracted/per_task/t02_declared_probe.csv \\
        --output-prefix cronbach_alpha_task4

Outputs
-------
    {prefix}.txt  Human-readable summary
    {prefix}.csv  Per-domain x per-threshold rows

Dependencies
------------
    numpy, pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = [23, 20, 15]  # include an item if at least t of 23 models scored it


def cronbach_alpha(item_respondent: np.ndarray) -> tuple[float, int, int]:
    """Compute Cronbach's alpha from a strictly complete items x respondents
    matrix. Returns (alpha, n_items, n_respondents). NaN if degenerate."""
    M = np.asarray(item_respondent, dtype=float)
    if M.size == 0 or np.isnan(M).any():
        return (np.nan, int(M.shape[0]) if M.size else 0,
                int(M.shape[1]) if M.size else 0)
    k, n = M.shape
    if k < 2 or n < 2:
        return (np.nan, k, n)
    item_vars = M.var(axis=1, ddof=1)
    total = M.sum(axis=0)
    total_var = total.var(ddof=1)
    if total_var == 0:
        return (np.nan, k, n)
    return (float(k / (k - 1.0) * (1.0 - item_vars.sum() / total_var)), k, n)


def alpha_at_threshold(wide: pd.DataFrame, t: int) -> tuple[float, int, int]:
    """Keep items (rows) scored by at least t respondents; then drop any
    respondents (cols) with NaN on the surviving item set. Returns alpha on
    the resulting strictly complete matrix."""
    per_item = wide.notna().sum(axis=1)
    kept_items = wide.index[per_item >= t]
    sub = wide.loc[kept_items]
    kept_resp = sub.columns[sub.notna().all(axis=0)]
    sub = sub.loc[:, kept_resp]
    return cronbach_alpha(sub.values)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task4-csv", type=Path, required=True,
                    help="per_task/t04_confidence_entropy.csv from extractor")
    ap.add_argument("--task2-csv", type=Path, default=None,
                    help="per_task/t02_declared_probe.csv, for category join "
                         "(optional but recommended for per-domain output)")
    ap.add_argument("--score-col", type=str, default="score",
                    help="Column from Task 4 CSV to use as per-trial score")
    ap.add_argument("--output-prefix", type=str, default="cronbach_alpha_task4")
    args = ap.parse_args()

    df = pd.read_csv(args.task4_csv)
    if args.score_col not in df.columns:
        raise ValueError(f"Score column {args.score_col!r} not in Task 4 CSV. "
                         f"Available: {list(df.columns)}")
    print(f"Using score column: {args.score_col!r}")
    print(f"Task 4: {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['question_id'].nunique()} questions")

    # Join categories from Task 2 if provided
    if args.task2_csv is not None and args.task2_csv.exists():
        t2 = pd.read_csv(args.task2_csv)
        qcat = t2[["question_id", "category"]].drop_duplicates("question_id")
        df = df.merge(qcat, on="question_id", how="left")
        print(f"  Joined categories from Task 2: "
              f"{df['category'].nunique()} unique categories")
    else:
        df["category"] = "ALL"
        print("  No Task 2 CSV provided; per-domain stratification skipped.")

    # Build items x respondents wide matrix (rows = question_id, cols = model)
    wide = df.pivot_table(
        index="question_id", columns="model",
        values=args.score_col, aggfunc="first"
    )
    print(f"  Wide matrix: {wide.shape[0]} items x {wide.shape[1]} models")
    print(f"  Completion: {wide.notna().sum().sum()} / {wide.size} cells present")

    rows = []
    print("\nPooled alpha by inclusion threshold:")
    for t in THRESHOLDS:
        a, k, n = alpha_at_threshold(wide, t)
        a_str = f"{a:.3f}" if not np.isnan(a) else "nan"
        print(f"  threshold >= {t:2d} models  ->  k={k:>3}, n={n:>3}, alpha={a_str}")
        rows.append({
            "domain":    "ALL",
            "threshold": t,
            "n_items":   k,
            "n_models":  n,
            "alpha":     a,
        })

    if "category" in df.columns and df["category"].nunique() > 1:
        print("\nPer-domain alpha (at threshold = 15):")
        for cat, sub in df.groupby("category"):
            wide_sub = sub.pivot_table(
                index="question_id", columns="model",
                values=args.score_col, aggfunc="first",
            )
            for t in THRESHOLDS:
                a, k, n = alpha_at_threshold(wide_sub, t)
                rows.append({
                    "domain":    str(cat),
                    "threshold": t,
                    "n_items":   k,
                    "n_models":  n,
                    "alpha":     a,
                })
                if t == 15:
                    a_str = f"{a:.3f}" if not np.isnan(a) else "nan"
                    print(f"  {cat:<40}  k={k:>3}, n={n:>3}, alpha={a_str}")

    results = pd.DataFrame(rows)
    out_csv = Path(f"{args.output_prefix}.csv")
    out_txt = Path(f"{args.output_prefix}.txt")
    results.to_csv(out_csv, index=False)

    with out_txt.open("w") as f:
        f.write("Cronbach's alpha for Task 4 (Confidence Entropy)\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Score column: {args.score_col}\n")
        f.write(f"Input: {len(df)} rows, {df['model'].nunique()} models, "
                f"{df['question_id'].nunique()} questions\n\n")
        f.write("Pooled alpha by inclusion threshold:\n")
        for r in results[results["domain"] == "ALL"].itertuples():
            a = f"{r.alpha:.3f}" if not np.isnan(r.alpha) else "nan"
            f.write(f"  threshold >= {r.threshold:>2} models  "
                    f"k={r.n_items:>3}  n={r.n_models:>3}  alpha={a}\n")
        f.write("\n")
        if results["domain"].nunique() > 1:
            f.write("Per-domain alpha (threshold = 15):\n")
            for r in results[(results["domain"] != "ALL") &
                             (results["threshold"] == 15)].itertuples():
                a = f"{r.alpha:.3f}" if not np.isnan(r.alpha) else "nan"
                f.write(f"  {r.domain:<45}  k={r.n_items:>3}  "
                        f"n={r.n_models:>3}  alpha={a}\n")

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
