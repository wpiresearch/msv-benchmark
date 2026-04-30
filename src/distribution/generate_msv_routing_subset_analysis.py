#!/usr/bin/env python3
"""
generate_msv_routing_subset_analysis.py
========================================

Regenerates the `msv_routing_subset_analysis.{pdf,png,csv}` figure used
in Appendix D of the paper (Figure 1).

What it computes
----------------
For each of the 31 non-empty subsets of the five MSV dimensions
{CE, ER, CI, EM, PI}, we:
  1. Compute each model's mean declared activation on that subset
     (from Task 2 declared-probe outputs).
  2. Rank models by this activation score.
  3. Correlate the resulting ranking with observed Task 1 (Delegate Game)
     behavioral scores via Spearman's rho.

The CE-only baseline corresponds to standard declarative calibration
evaluation. If CE alone is the right predictor, multi-dimensional subsets
should do no better. The empirical result (CE-only = worst single-
dimension predictor at rho = -0.315) supports the paper's multi-
dimensional thesis directly.

Inputs
------
  --task2-csv   Path to per_task/t02_declared_probe.csv from the
                extractor output (kaggle_extracted/). Must contain
                columns: model, declared_CE, declared_ER, declared_CI,
                declared_EM, declared_PI.
  --metadata-csv Path to run_metadata.csv from the extractor output.
                Must contain columns: model, task_id, run_result_value.
  --output-prefix  Basename (without extension) for the three output
                files. Default "msv_routing_subset_analysis".

Usage
-----
  python generate_msv_routing_subset_analysis.py \\
      --task2-csv    ./kaggle_extracted/per_task/t02_declared_probe.csv \\
      --metadata-csv ./kaggle_extracted/run_metadata.csv \\
      --output-prefix msv_routing_subset_analysis

Outputs
-------
  {prefix}.csv   One row per subset, columns:
                   subset, size, spearman_rho, p
  {prefix}.pdf   Horizontal bar chart of all 31 subsets, sorted by rho.
                 CE-only baseline highlighted in red. Bar color = subset
                 size.
  {prefix}.png   PNG version at 150 dpi for web preview.

Dependencies
------------
  numpy, pandas, scipy, matplotlib
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DIMS = ["CE", "ER", "CI", "EM", "PI"]


def load_declared_msv(task2_csv: Path) -> pd.DataFrame:
    """Load per-model mean declared MSV vector from Task 2 output.

    Task 2 has one row per (model, question) with columns declared_CE,
    declared_ER, declared_CI, declared_EM, declared_PI. We average over
    questions per model to obtain a single 5-vector per model.
    """
    df = pd.read_csv(task2_csv)
    cols = [f"declared_{d}" for d in DIMS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Task 2 CSV missing columns: {missing}")
    msv = df.groupby("model")[cols].mean()
    msv.columns = DIMS
    return msv


def load_task1_scores(metadata_csv: Path) -> pd.Series:
    """Load platform-authoritative Task 1 scores per model.

    `run_result_value` from run_metadata.csv is the Kaggle leaderboard
    score (mean over all scheduled trials, missing = 0).
    """
    md = pd.read_csv(metadata_csv)
    t1 = md[md["task_id"] == "t01"].set_index("model")["run_result_value"]
    t1 = pd.to_numeric(t1, errors="coerce")
    return t1


def compute_subset_correlations(msv: pd.DataFrame,
                                t1_scores: pd.Series) -> pd.DataFrame:
    """Compute Spearman rho between subset activation and Task 1 score.

    For each non-empty subset of DIMS, computes the mean of those
    declared values per model and correlates with T1 score.
    """
    common = msv.index.intersection(t1_scores.index)
    msv = msv.loc[common]
    t1_scores = t1_scores.loc[common]
    print(f"  Common models across MSV and T1: {len(common)}")

    subsets = []
    for k in range(1, len(DIMS) + 1):
        subsets.extend(combinations(DIMS, k))

    rows = []
    for subset in subsets:
        activation = msv[list(subset)].mean(axis=1)
        rho, p = stats.spearmanr(activation, t1_scores)
        rows.append({
            "subset":       "+".join(subset),
            "size":         len(subset),
            "spearman_rho": float(rho),
            "p":            float(p),
        })
    df = pd.DataFrame(rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    return df


def plot_subset_bar(df: pd.DataFrame, output_prefix: Path) -> None:
    """Horizontal bar chart of subsets, CE-only highlighted."""
    df_plot = df.sort_values("spearman_rho").reset_index(drop=True)
    colors = plt.cm.viridis((df_plot["size"] - 1) / 4.0)

    fig, ax = plt.subplots(figsize=(9, 10))
    bars = ax.barh(range(len(df_plot)), df_plot["spearman_rho"],
                   color=colors, edgecolor="black", linewidth=0.5)

    # Highlight CE-only baseline
    ce_only_idx_list = df_plot.index[df_plot["subset"] == "CE"].tolist()
    if ce_only_idx_list:
        ce_idx = ce_only_idx_list[0]
        bars[ce_idx].set_edgecolor("red")
        bars[ce_idx].set_linewidth(2.5)
        ce_rho = df_plot.loc[ce_idx, "spearman_rho"]
        ax.axvline(ce_rho, color="red", linestyle="--", linewidth=1,
                   label=f"CE-only baseline ($\\rho$={ce_rho:.2f})")
        ax.legend(loc="lower right", fontsize=9)

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot["subset"], fontsize=8, family="monospace")
    ax.set_xlabel("Spearman rho (subset activation vs. Task 1 behavioral score)")
    ax.set_title(
        "MSV dimension-subset routing quality (23-model Kaggle cohort)\n"
        "CE-only baseline outlined in red. Bar color = subset size."
    )
    ax.axvline(0, color="black", linewidth=0.5)

    # Colorbar for subset size
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=1, vmax=5))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.35, pad=0.02)
    cbar.set_label("Subset size", fontsize=9)
    cbar.set_ticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    pdf_path = Path(f"{output_prefix}.pdf")
    png_path = Path(f"{output_prefix}.png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wrote: {pdf_path}")
    print(f"  Wrote: {png_path}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task2-csv", type=Path, required=True,
                    help="per_task/t02_declared_probe.csv from the extractor")
    ap.add_argument("--metadata-csv", type=Path, required=True,
                    help="run_metadata.csv from the extractor")
    ap.add_argument("--output-prefix", type=str, default="msv_routing_subset_analysis",
                    help="Basename for output .csv, .pdf, and .png (no extension)")
    args = ap.parse_args()

    print("Loading Task 2 declared-MSV vectors...")
    msv = load_declared_msv(args.task2_csv)
    print(f"  {len(msv)} models with declared MSV vectors")

    print("Loading Task 1 platform-authoritative scores...")
    t1 = load_task1_scores(args.metadata_csv)
    print(f"  {len(t1)} models with Task 1 scores")

    print("Computing 31-subset correlations...")
    df = compute_subset_correlations(msv, t1)

    csv_path = Path(f"{args.output_prefix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Wrote: {csv_path}")

    print("\nTop 10 subsets by Spearman rho:")
    print(df.head(10).to_string(index=False))
    print("\nBottom 5 subsets:")
    print(df.tail(5).to_string(index=False))
    ce_row = df[df["subset"] == "CE"]
    if not ce_row.empty:
        ce_rho = ce_row["spearman_rho"].iloc[0]
        ce_rank = df.index[df["subset"] == "CE"][0] + 1
        print(f"\nCE-only baseline: rho = {ce_rho:.3f}, rank {ce_rank} of {len(df)}")

    print("\nGenerating figure...")
    plot_subset_bar(df, args.output_prefix)
    print("\nDone.")


if __name__ == "__main__":
    main()
