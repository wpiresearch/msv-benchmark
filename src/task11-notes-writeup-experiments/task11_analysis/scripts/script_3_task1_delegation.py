"""
Script 3: Task 1 delegation-by-difficulty curves.

Task 1 (Delegate Game) rewards DELEGATE on hard questions and penalizes it
on easy ones. A metacognitively functional model should show a monotonically
increasing delegation rate as empirical difficulty rises. A flat near-zero
curve indicates the model answers everything regardless of difficulty --
suggesting its routing signal carries no useful metacognitive information.

This script computes per-model delegation rates in difficulty bins and
produces a figure that contrasts reasoning models (expected to be flat)
against non-reasoning models (expected to show positive slope).

Expected CSV columns for Task 1:
  - question_id
  - choice: "ANSWER" or "DELEGATE"
  - answer: letter or None
  - confidence: int 1-4 or None
  - correct: correct answer letter
  - difficulty: float in [0, 1]
  - score: float in [0, 1]

Inputs:
  --catalog: path to run_catalog.csv
  --task-id: default "t01"
  --n-bins: difficulty bins (default 5)
  --out-csv: path to write task1_delegation_curves.csv
  --out-fig: path to write delegation_by_difficulty.png

Outputs:
  task1_delegation_curves.csv: one row per (model, difficulty_bin) with
    n_questions, delegate_rate, mean_score, difficulty_bin_mid.

  delegation_by_difficulty.png: line plot, one line per model, x-axis is
    difficulty bin midpoint, y-axis is delegation rate. Reasoning models
    colored red, non-reasoning models colored blue, alpha adjusted for
    visibility when many models are shown.

Usage:
  python script_3_task1_delegation.py \\
      --catalog outputs/run_catalog.csv \\
      --out-csv outputs/task1_delegation_curves.csv \\
      --out-fig outputs/delegation_by_difficulty.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def compute_delegation_curve(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """Bin by difficulty and compute delegation rate per bin."""
    df = df.copy()
    df = df.dropna(subset=["difficulty", "choice"])
    if len(df) == 0:
        return pd.DataFrame()

    # Use fixed-edge bins from 0 to 1 so curves from different models are
    # comparable. Quantile binning would give each model its own edges.
    edges = np.linspace(df["difficulty"].min(), df["difficulty"].max(), n_bins + 1)
    # Ensure strictly increasing edges (guard against constant difficulty)
    if edges[-1] <= edges[0]:
        edges = np.linspace(0, 1, n_bins + 1)
    df["diff_bin"] = pd.cut(df["difficulty"], bins=edges,
                            include_lowest=True, labels=False)

    agg = df.groupby("diff_bin").agg(
        n_questions=("question_id", "count"),
        delegate_rate=("choice", lambda x: (x == "DELEGATE").mean()),
        answer_rate=("choice", lambda x: (x == "ANSWER").mean()),
        mean_score=("score", "mean"),
        mean_difficulty=("difficulty", "mean"),
    ).reset_index()
    agg["delegate_rate"] = agg["delegate_rate"].round(4)
    agg["mean_score"] = agg["mean_score"].round(4)
    agg["mean_difficulty"] = agg["mean_difficulty"].round(4)
    return agg


def compute_slope(curve: pd.DataFrame) -> float:
    """Least-squares slope of delegate_rate vs mean_difficulty."""
    if len(curve) < 2 or curve["delegate_rate"].nunique() < 2:
        return 0.0
    x = curve["mean_difficulty"].values
    y = curve["delegate_rate"].values
    # Guard against constant x
    if np.std(x) < 1e-9:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def build_delegation_figure(per_model_curves: dict[str, pd.DataFrame],
                            is_reasoning: dict[str, bool],
                            out_path: Path) -> None:
    """Plot delegation-by-difficulty curves, colored by model category."""
    fig, ax = plt.subplots(figsize=(10, 7))

    reasoning_colors = plt.get_cmap("Reds")(np.linspace(0.45, 0.95, 10))
    non_reasoning_colors = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, 10))

    r_idx, n_idx = 0, 0
    for model, curve in per_model_curves.items():
        if curve.empty:
            continue
        if is_reasoning.get(model, False):
            color = reasoning_colors[r_idx % len(reasoning_colors)]
            r_idx += 1
            ls = "--"
        else:
            color = non_reasoning_colors[n_idx % len(non_reasoning_colors)]
            n_idx += 1
            ls = "-"

        ax.plot(curve["mean_difficulty"], curve["delegate_rate"],
                marker="o", linestyle=ls, color=color, label=model, alpha=0.85)

    ax.set_xlabel("Question difficulty (bin midpoint)")
    ax.set_ylabel("Delegation rate")
    ax.set_title("Task 1 delegation rate as a function of difficulty\n"
                 "Solid = non-reasoning (expect positive slope); "
                 "dashed = reasoning (expect flat)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--task-id", default="t01")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    catalog = pd.read_csv(args.catalog)
    task_rows = catalog[catalog["task_id"] == args.task_id]

    all_curves = []
    per_model_curves: dict[str, pd.DataFrame] = {}
    is_reasoning: dict[str, bool] = {}
    slopes = []

    for _, cat_row in task_rows.iterrows():
        csv_path = cat_row.get("csv_file_path")
        if not isinstance(csv_path, str) or not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {cat_row['model']}: {e}")
            continue
        if "difficulty" not in df.columns or "choice" not in df.columns:
            print(f"Skipping {cat_row['model']}: missing difficulty/choice columns")
            continue

        curve = compute_delegation_curve(df, n_bins=args.n_bins)
        if curve.empty:
            continue
        curve.insert(0, "model", cat_row["model"])
        curve["is_reasoning"] = cat_row["is_reasoning"]
        all_curves.append(curve)
        per_model_curves[cat_row["model"]] = curve
        is_reasoning[cat_row["model"]] = bool(cat_row["is_reasoning"])
        slopes.append({
            "model": cat_row["model"],
            "is_reasoning": bool(cat_row["is_reasoning"]),
            "delegation_slope": round(compute_slope(curve), 4),
            "overall_delegate_rate": round(
                (df["choice"] == "DELEGATE").mean(), 4),
            "n_questions": len(df),
        })

    if not all_curves:
        print(f"No {args.task_id} data could be analyzed.")
        return

    combined = pd.concat(all_curves, ignore_index=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    # Also save slopes summary
    slopes_df = pd.DataFrame(slopes).sort_values("delegation_slope")
    slopes_path = args.out_csv.with_name(args.out_csv.stem + "_slopes.csv")
    slopes_df.to_csv(slopes_path, index=False)

    print(f"Computed curves for {len(per_model_curves)} models on {args.task_id}.")
    print("\nPer-model slopes (delegation rate vs difficulty):")
    print(slopes_df.to_string(index=False))

    build_delegation_figure(per_model_curves, is_reasoning, args.out_fig)
    print(f"\nCurves CSV: {args.out_csv}")
    print(f"Slopes CSV: {slopes_path}")
    print(f"Figure:     {args.out_fig}")


if __name__ == "__main__":
    main()
