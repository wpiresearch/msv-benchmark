#!/usr/bin/env python3
"""
compute_cronbach_alpha_all_tasks.py
====================================

Generalized Cronbach's alpha computation for the MSV Benchmark task suite.
Adopts the same balance conventions and per-domain stratification used by
the Task 4 script (compute_cronbach_alpha_task4.py) and applies them
uniformly to all tasks that have a per-trial score.

Rationale
---------
Section 6.2 limitation 10 currently states "Cronbach's alpha for Task 4
... alpha for the remaining tasks is pending." Three concurrent benchmarks
(Cacioli 2026, Singh 2026, Parikh 2026) report per-task internal-consistency
reliability via Cronbach's alpha; computing the same statistic for our
remaining tasks closes the psychometric reporting gap.

Per-task conventions
--------------------
- Task 1 (Delegate Game):       use 'score' (the reward 0.00-1.00).
- Task 2 (Declared MSV Probe):  use 'routing_score' (the principal Task 2
                                metric: combines parseability,
                                differentiation, and routing alignment).
- Task 3 (Second-Chance):       use 'score' (the per-trial reward).
- Task 4 (Confidence Entropy):  reproduces the existing Task 4 script's
                                values; included for cross-checking.
- Task 5 (Teammate Delegate):   use 'score'.
- Task 6 (Behavioral ER):       use 'score'.
- Task 7 (Behavioral CI):       use 'score'.
- Task 8 (Behavioral EM):       use 'score'. NOTE: indexed by 'id'
                                rather than 'question_id'.
- Task 9 (Behavioral PI):       use 'score'.
- Task 10 (DPP Sequence):       use 'score'. Cohort heavily incomplete on
                                Kaggle (1/23 clean); reported with caveats.
- Task 11 (MC Binary Pairs):    use 'judgment_correct' aggregated to
                                per-question accuracy (mean of signal and
                                lure trials). Confidence is reported
                                separately as a sensitivity column.

Three balance conventions
-------------------------
Threshold t = 23 (strict balanced panel), 20, and 15. At each threshold,
keep only items completed by >= t models, then drop any models that have
NaN on the surviving items. Compute alpha on the resulting strictly
complete matrix.

Per-domain stratification
-------------------------
GPQA category (biology, chemistry, physics) joined from Task 2 where
question_id is available. Tasks indexed by something other than the
GPQA question_id (e.g., Task 8's pair ID) are reported pooled-only.

Usage
-----
    python compute_cronbach_alpha_all_tasks.py \\
        --per-task-dir data/kaggle-data/kaggle_extracted/per_task/ \\
        --output-prefix cronbach_alpha_all_tasks

Outputs
-------
    {prefix}.csv  Wide table: task x threshold x domain -> alpha
    {prefix}.txt  Human-readable summary; reproduces the table format
                  the paper uses.

Dependencies
------------
    numpy, pandas
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


THRESHOLDS = [23, 20, 15]
N_MODELS_KAGGLE = 23  # cohort size

# Per-task config: (filename_stem, score_column, item_id_column, label)
TASK_CONFIG = [
    ("t01_delegate_game",       "score",            "question_id", "Task 1 (Delegate Game)"),
    ("t02_declared_probe",      "routing_score",    "question_id", "Task 2 (Declared MSV Probe)"),
    ("t03_second_chance",       "score",            "question_id", "Task 3 (Second-Chance)"),
    ("t04_confidence_entropy",  "score",            "question_id", "Task 4 (Confidence Entropy)"),
    ("t05_teammate_delegate",   "score",            "question_id", "Task 5 (Teammate Delegate)"),
    ("t06_behavioral_er",       "score",            "question_id", "Task 6 (Behavioral ER)"),
    ("t07_behavioral_ci",       "score",            "question_id", "Task 7 (Behavioral CI)"),
    ("t08_behavioral_em",       "score",            "id",          "Task 8 (Behavioral EM)"),
    ("t09_behavioral_pi",       "score",            "question_id", "Task 9 (Behavioral PI)"),
    ("t10_dpp_sequence",        "score",            "question_id", "Task 10 (DPP Sequence)"),
    ("t11_mc_binary_pairs",     "judgment_correct", "question_id", "Task 11 (MC Binary Pairs)"),
]


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
    """Keep items scored by at least t respondents; drop respondents with
    NaN on that surviving item set; compute alpha."""
    per_item = wide.notna().sum(axis=1)
    kept_items = wide.index[per_item >= t]
    sub = wide.loc[kept_items]
    if sub.empty:
        return (np.nan, 0, 0)
    kept_resp = sub.columns[sub.notna().all(axis=0)]
    sub = sub.loc[:, kept_resp]
    return cronbach_alpha(sub.values)


def build_wide(df: pd.DataFrame, item_col: str, score_col: str) -> pd.DataFrame:
    """Items x respondents matrix. For Task 11, score_col may be a
    pre-aggregated per-question accuracy (mean of two trials)."""
    return df.pivot_table(
        index=item_col, columns="model", values=score_col, aggfunc="first"
    )


def prepare_task_df(stem: str, score_col: str, item_col: str,
                    csv_path: Path, t2_categories: pd.DataFrame | None) -> pd.DataFrame:
    """Load and lightly preprocess a per-task CSV. Task 11 needs aggregation
    across signal/lure pairs; other tasks pass through."""
    df = pd.read_csv(csv_path)
    if score_col not in df.columns:
        raise ValueError(
            f"[{stem}] score column {score_col!r} not present. "
            f"Available: {list(df.columns)}"
        )
    if item_col not in df.columns:
        raise ValueError(
            f"[{stem}] item id column {item_col!r} not present. "
            f"Available: {list(df.columns)}"
        )
    if stem == "t11_mc_binary_pairs":
        # Aggregate the 2 trials per question (signal + lure) into a per-
        # question accuracy. judgment_correct is binary; mean over the 2
        # rows produces 0.0, 0.5, or 1.0.
        agg = df.groupby(["model", item_col], as_index=False)[score_col].mean()
        df = agg
    if t2_categories is not None and item_col == "question_id":
        df = df.merge(t2_categories, on="question_id", how="left")
    else:
        df["category"] = "ALL"
    return df


def analyze_task(stem: str, score_col: str, item_col: str, label: str,
                 csv_path: Path, t2_categories: pd.DataFrame | None) -> list[dict]:
    """Run pooled and per-domain alpha for one task. Returns a list of row
    dicts suitable for concatenation into a DataFrame."""
    if not csv_path.exists():
        print(f"[{stem}] CSV not found at {csv_path}; skipping")
        return []
    df = prepare_task_df(stem, score_col, item_col, csv_path, t2_categories)
    n_models = df["model"].nunique()
    n_items = df[item_col].nunique()
    n_rows = len(df)
    print(f"\n=== {label} ===")
    print(f"  Source: {csv_path.name}  ({n_rows} rows, "
          f"{n_models} models, {n_items} items)")

    rows = []
    wide = build_wide(df, item_col, score_col)
    completion_rate = wide.notna().sum().sum() / wide.size
    print(f"  Wide matrix: {wide.shape[0]} items x {wide.shape[1]} models  "
          f"({completion_rate:.1%} cells present)")

    print("  Pooled alpha:")
    for t in THRESHOLDS:
        a, k, n = alpha_at_threshold(wide, t)
        a_str = f"{a:.3f}" if not np.isnan(a) else "nan"
        print(f"    threshold >= {t:2d} models  ->  k={k:>3}, n={n:>3}, alpha={a_str}")
        rows.append({
            "task":      label,
            "task_stem": stem,
            "domain":    "ALL",
            "threshold": t,
            "n_items":   k,
            "n_models":  n,
            "alpha":     a,
        })

    if "category" in df.columns and df["category"].nunique() > 1:
        print("  Per-domain alpha (threshold = 15):")
        for cat, sub in df.groupby("category"):
            wide_sub = build_wide(sub, item_col, score_col)
            for t in THRESHOLDS:
                a, k, n = alpha_at_threshold(wide_sub, t)
                rows.append({
                    "task":      label,
                    "task_stem": stem,
                    "domain":    str(cat),
                    "threshold": t,
                    "n_items":   k,
                    "n_models":  n,
                    "alpha":     a,
                })
                if t == 15:
                    a_str = f"{a:.3f}" if not np.isnan(a) else "nan"
                    print(f"    {cat:<40}  k={k:>3}, n={n:>3}, alpha={a_str}")
    else:
        print("  (no per-domain stratification: item key is not GPQA question_id "
              "or only one category present)")

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--per-task-dir", type=Path, required=True,
                    help="Directory containing per_task/t??_*.csv files")
    ap.add_argument("--output-prefix", type=str, default="cronbach_alpha_all_tasks",
                    help="Output file prefix (default: cronbach_alpha_all_tasks)")
    args = ap.parse_args()

    if not args.per_task_dir.is_dir():
        print(f"ERROR: {args.per_task_dir} is not a directory")
        return 1

    # Pre-load Task 2 categories for joining
    t2_path = args.per_task_dir / "t02_declared_probe.csv"
    if t2_path.exists():
        t2 = pd.read_csv(t2_path)
        t2_categories = t2[["question_id", "category"]].drop_duplicates("question_id")
        print(f"Loaded category map from t02_declared_probe.csv: "
              f"{t2_categories['category'].nunique()} categories, "
              f"{len(t2_categories)} questions")
    else:
        t2_categories = None
        print("WARNING: t02_declared_probe.csv not found; "
              "per-domain analysis disabled")

    all_rows = []
    for stem, score_col, item_col, label in TASK_CONFIG:
        csv_path = args.per_task_dir / f"{stem}.csv"
        rows = analyze_task(stem, score_col, item_col, label, csv_path, t2_categories)
        all_rows.extend(rows)

    # Write outputs
    df_out = pd.DataFrame(all_rows)
    out_csv = Path(f"{args.output_prefix}.csv")
    out_txt = Path(f"{args.output_prefix}.txt")
    df_out.to_csv(out_csv, index=False)

    with out_txt.open("w") as f:
        f.write("Cronbach's alpha across the MSV Benchmark task suite\n")
        f.write("=" * 60 + "\n\n")
        f.write("Convention: items x respondents matrix is questions x models;\n")
        f.write("alpha computed at threshold t = number of models that completed\n")
        f.write("the item. THRESHOLDS = [23 (strict), 20, 15].\n\n")
        f.write("Pooled alpha by task and threshold:\n")
        f.write("-" * 60 + "\n")
        for label in [t[3] for t in TASK_CONFIG]:
            sub = df_out[(df_out["task"] == label) & (df_out["domain"] == "ALL")]
            if sub.empty:
                continue
            f.write(f"\n{label}\n")
            for r in sub.itertuples():
                a = f"{r.alpha:.3f}" if not np.isnan(r.alpha) else "nan"
                f.write(f"  threshold >= {r.threshold:>2}  "
                        f"k={r.n_items:>3}  n={r.n_models:>3}  alpha={a}\n")
        f.write("\n")
        f.write("Per-domain alpha (threshold = 15):\n")
        f.write("-" * 60 + "\n")
        for label in [t[3] for t in TASK_CONFIG]:
            sub = df_out[(df_out["task"] == label) &
                         (df_out["domain"] != "ALL") &
                         (df_out["threshold"] == 15)]
            if sub.empty:
                continue
            f.write(f"\n{label}\n")
            for r in sub.itertuples():
                a = f"{r.alpha:.3f}" if not np.isnan(r.alpha) else "nan"
                f.write(f"  {r.domain:<45}  k={r.n_items:>3}  "
                        f"n={r.n_models:>3}  alpha={a}\n")

    print(f"\nSaved: {out_csv}")
    print(f"Saved: {out_txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
