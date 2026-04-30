#!/usr/bin/env python3
"""
generate_rank_reversal_figure.py
=================================

Generates the main-text rank-reversal figure that visualizes the central
claim of the paper: declarative calibration quality (ECE) and behavioral
routing quality (Delegation AUC-ROC) induce different model rankings on
the same models and same items.

The figure is two panels side-by-side:
  Left  : 6-model Turing cohort (from Table 3 in the paper)
  Right : 23-model Kaggle cohort (computable subset, from Turn 2 analysis)

Each panel plots ECE rank (x-axis, 1 = best declarative calibration) vs.
Delegation AUC-ROC rank (y-axis, 1 = best behavioral routing) with model
labels. Points near the diagonal agree on both dimensions; points far
from the diagonal show rank reversal. The highlighted examples in the
paper's text (llama3.2:1b for Turing, qwen3-235b-a22b-instruct-2507 and
claude-haiku-4-5 for Kaggle) are labeled prominently.

Usage
-----
  python generate_rank_reversal_figure.py \\
      --turing-csv   turing_comparative_table3.csv \\
      --kaggle-csv   kaggle_analysis/comparative/delegate_game_metrics.csv \\
      --output-prefix rank_reversal_scatter

Inputs
------
  --turing-csv : six-row CSV with columns model, ece, deleg_auc
                 (built from Table 3 in main-ed.tex).
  --kaggle-csv : the delegate_game_metrics.csv produced by the
                 Turn 2 analysis (kaggle_analysis/comparative/).

Outputs
-------
  {prefix}.pdf  Main-text figure
  {prefix}.png  Web preview at 150 dpi

Dependencies
------------
  numpy, pandas, matplotlib
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def rank_reversal_panel(ax, df, title, highlight_models=None,
                        label_threshold_diff=None):
    """Plot ECE-rank vs Delegation-AUC-rank on one axes."""
    n = len(df)
    df = df.copy()
    df["ece_rank"]   = df["ece"].rank(ascending=True,  method="min")
    df["deleg_rank"] = df["deleg_auc"].rank(ascending=False, method="min")
    df["rank_diff"]  = (df["ece_rank"] - df["deleg_rank"]).abs()

    # Diagonal reference line
    ax.plot([0.5, n + 0.5], [0.5, n + 0.5], color="gray",
            linestyle="--", linewidth=1, alpha=0.5,
            label="agreement diagonal")

    # Points: color by rank difference
    sizes = 60 + 20 * df["rank_diff"]
    sc = ax.scatter(df["ece_rank"], df["deleg_rank"],
                    c=df["rank_diff"], cmap="plasma_r",
                    s=sizes, edgecolor="black", linewidth=0.7, zorder=3,
                    vmin=0, vmax=max(df["rank_diff"].max(), 3))

    # Hand-tuned label offsets (dx, dy, ha, va) per model to avoid collisions.
    LABEL_OFFSETS = {
        # Turing cohort
        "mistral:7b":             ( 10,  0, "left",  "center"),
        "gemma2:9b":              ( 10,  0, "left",  "center"),
        "llama3.1:8b":            (  0,  12, "center", "bottom"),
        "phi4-mini":              (  0,  12, "center", "bottom"),
        "qwen2.5:7b":             ( 10,  0, "left",  "center"),
        "llama3.2:1b":            (  0,  12, "center", "bottom"),
        # Kaggle cohort
        "claude-haiku-4.5":       ( 10,  0, "left",  "center"),
        "qwen3-235b-a22b":        ( 10,  0, "left",  "center"),
        "gpt-5.4-nano":           ( 10,  0, "left",  "center"),
        "gemini-2.0-flash-lite":  ( 10,  0, "left",  "center"),
        "gpt-oss-20b":            ( 10,  0, "left",  "center"),
        "claude-opus-4.1":        (-10,  0, "right", "center"),
        "claude-opus-4.6":        (-10,  0, "right", "center"),
        "gpt-5.4":                (-10,  0, "right", "center"),
        "gemma-3-27b-it":         ( 10,  0, "left",  "center"),
        "gemini-3.1-pro":         (-10,  0, "right", "center"),
    }

    if label_threshold_diff is None:
        to_label = df
    else:
        to_label = df[df["rank_diff"] >= label_threshold_diff]

    for _, r in to_label.iterrows():
        is_highlight = highlight_models and r["model"] in highlight_models
        dx, dy, ha, va = LABEL_OFFSETS.get(r["model"],
                                           (8, 8, "left", "bottom"))
        ax.annotate(
            r["model"],
            (r["ece_rank"], r["deleg_rank"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7.5 if not is_highlight else 9,
            fontweight="bold" if is_highlight else "normal",
            color="#b22222" if is_highlight else "black",
            ha=ha, va=va,
        )

    ax.set_xlabel("ECE rank (1 = best declarative calibration)")
    ax.set_ylabel("Delegation AUC-ROC rank (1 = best behavioral routing)")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(0.3, n + 0.7)
    ax.set_xticks(range(1, n + 1))
    ax.set_yticks(range(1, n + 1))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, zorder=0)
    ax.invert_yaxis()  # rank 1 at top
    ax.invert_xaxis()  # rank 1 at left

    return sc


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--turing-csv", type=Path, required=True)
    ap.add_argument("--kaggle-csv", type=Path, required=True)
    ap.add_argument("--output-prefix", type=str, default="rank_reversal_scatter")
    args = ap.parse_args()

    # --- Load Turing cohort (from the paper's Table 3) ---
    turing = pd.read_csv(args.turing_csv)
    required_cols = {"model", "ece", "deleg_auc"}
    if not required_cols.issubset(turing.columns):
        raise ValueError(f"Turing CSV missing columns {required_cols}")
    print(f"Turing: {len(turing)} models")

    # --- Load and filter Kaggle cohort ---
    kaggle_raw = pd.read_csv(args.kaggle_csv)
    # The input CSV must have columns ece_4bin and deleg_auc_vs_own_err
    # (the latter is derived from the raw per-trial data; see the
    # rank_divergence audit for why this is the correct AUC target).
    # Drop models where either metric is undefined.
    kaggle = kaggle_raw.dropna(subset=["deleg_auc_vs_own_err", "ece_4bin"]).copy()
    kaggle = kaggle.rename(columns={
        "ece_4bin":              "ece",
        "deleg_auc_vs_own_err":  "deleg_auc",
    })[["model", "ece", "deleg_auc"]]
    # Shorten long model names for display
    display_rename = {
        "qwen3-235b-a22b-instruct-2507": "qwen3-235b-a22b",
        "qwen3-coder-480b-a35b-instruct": "qwen3-coder-480b",
        "qwen3-next-80b-a3b-instruct":   "qwen3-next-80b-inst",
        "qwen3-next-80b-a3b-thinking":   "qwen3-next-80b-think",
        "claude-haiku-4-5-20251001":     "claude-haiku-4.5",
        "claude-opus-4-1-20250805":      "claude-opus-4.1",
        "claude-opus-4-6-default":       "claude-opus-4.6",
        "gpt-5.4-2026-03-05":            "gpt-5.4",
        "gpt-5.4-mini-2026-03-17":       "gpt-5.4-mini",
        "gpt-5.4-nano-2026-03-17":       "gpt-5.4-nano",
        "gemini-2.0-flash-lite-001":     "gemini-2.0-flash-lite",
        "gemini-3.1-pro-preview":        "gemini-3.1-pro",
    }
    kaggle["model"] = kaggle["model"].replace(display_rename)
    print(f"Kaggle (computable subset): {len(kaggle)} models")

    # --- Set up figure ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3),
                             gridspec_kw={"wspace": 0.4})

    turing_highlights = {"llama3.2:1b", "llama3.1:8b", "mistral:7b"}
    kaggle_highlights = {"qwen3-235b-a22b", "claude-haiku-4.5", "gemma-3-27b-it"}

    sc1 = rank_reversal_panel(
        axes[0], turing,
        f"Turing HPC cohort (n={len(turing)} open-weight models)",
        highlight_models=turing_highlights,
    )
    sc2 = rank_reversal_panel(
        axes[1], kaggle,
        f"Kaggle Benchmarks cohort (n={len(kaggle)} frontier + mid-tier)",
        highlight_models=kaggle_highlights,
        label_threshold_diff=0,  # label all on Kaggle panel
    )

    # Shared colorbar for rank-diff
    cbar = fig.colorbar(sc2, ax=axes, shrink=0.6, pad=0.02,
                        label="|ECE rank − Delegation-AUC rank|")
    # Also a legend note on the left for the diagonal
    axes[0].legend(loc="upper left", fontsize=8, framealpha=0.9)

    fig.suptitle(
        "Rank reversal: declarative calibration (ECE) vs. behavioral routing "
        "(Delegation AUC-ROC) produce different model rankings on the same "
        "items",
        fontsize=12.5, y=1.00,
    )

    pdf_path = Path(f"{args.output_prefix}.pdf")
    png_path = Path(f"{args.output_prefix}.png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote: {pdf_path}")
    print(f"Wrote: {png_path}")

    # Print rank-reversal summary for the caption
    print("\n--- Turing rank-reversals ---")
    t_sorted = turing.copy()
    t_sorted["ece_rank"] = t_sorted["ece"].rank(ascending=True, method="min")
    t_sorted["deleg_rank"] = t_sorted["deleg_auc"].rank(ascending=False, method="min")
    t_sorted["diff"] = (t_sorted["ece_rank"] - t_sorted["deleg_rank"]).abs()
    print(t_sorted.sort_values("diff", ascending=False).to_string(index=False))

    print("\n--- Kaggle rank-reversals ---")
    k_sorted = kaggle.copy()
    k_sorted["ece_rank"] = k_sorted["ece"].rank(ascending=True, method="min")
    k_sorted["deleg_rank"] = k_sorted["deleg_auc"].rank(ascending=False, method="min")
    k_sorted["diff"] = (k_sorted["ece_rank"] - k_sorted["deleg_rank"]).abs()
    print(k_sorted.sort_values("diff", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
