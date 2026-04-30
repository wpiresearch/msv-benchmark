#!/usr/bin/env python3
"""
generate_ece_vs_delegauc_scatter.py
====================================

Generates a single scatterplot of ECE (x-axis) vs. Delegation AUC-ROC
(y-axis) for every plottable model in the Kaggle cohort, with optional
overlay of the institutional cohort. Companion to Figure 1 of the
paper: Figure 1 shows the *rank* dissociation, this scatter shows the
underlying *raw* values, making the multi-quadrant pattern visible at
a glance (well-calibrated/good-routing, well-calibrated/poor-routing,
poor-calibrated/good-routing).

If declarative calibration and behavioral routing were redundant
metrics, we would expect a tight monotone relationship. The actual
pattern fills three quadrants plus near-chance models in the bottom
left, which is the visual analogue of the moderate-positive but
non-redundant Kendall tau = +0.45 reported in Section 5.2.

Inputs
------
- results/kaggle_cohort/comparative/delegate_game_metrics.csv
  (per-model ece_4bin, deleg_auc_vs_own_err, delegation_rate, n_trials,
  produced by analyze_kaggle_cohort.py)
- Optionally: an institutional comparative CSV with the same schema
  (e.g., the FA-protocol same-item comparison from Table 3 of the
  paper). If absent, plot is Kaggle-only.

ECE convention
--------------
Label-grouped binning (one bin per discrete confidence level), confidence
mapping {1,2,3,4} -> {0.25, 0.50, 0.75, 1.00}. The Kaggle cohort uses
answered-conditional ECE (non-delegated trials only); the institutional
overlay (when supplied) uses forced-answer ECE (full-panel calibration,
independent of delegation policy). This matches the paper's
estimand-distinct presentation in Tables 2 and 3.

Filtering
---------
A model is plottable iff Delegation AUC against own error is computable.
Models with delegation rate of 0% or 100%, or with all-correct or all-
wrong answered subsets, are excluded automatically because their
deleg_auc_vs_own_err column is NaN in the source CSV.

Usage
-----
    python generate_ece_vs_delegauc_scatter.py \\
        --kaggle-csv  results/kaggle_cohort/comparative/delegate_game_metrics.csv \\
        --output-png  results/figures/ece_vs_delegauc_scatter.png \\
        --output-pdf  results/figures/ece_vs_delegauc_scatter.pdf \\
        --output-csv  results/figures/ece_vs_delegauc_scatter.csv

With institutional overlay:

    python generate_ece_vs_delegauc_scatter.py \\
        --kaggle-csv         results/kaggle_cohort/comparative/delegate_game_metrics.csv \\
        --institutional-csv  results/turing_cohort/comparative_table_fa.csv \\
        --output-png         results/figures/ece_vs_delegauc_scatter.png \\
        --output-pdf         results/figures/ece_vs_delegauc_scatter.pdf \\
        --output-csv         results/figures/ece_vs_delegauc_scatter.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map column names from either the Kaggle metrics CSV or an
    institutional comparative table to a uniform schema."""
    aliases = {
        "ece_4bin":              "ece",
        "fa_ece":                "ece",
        "FA_ECE":                "ece",
        "ece":                   "ece",
        "deleg_auc_vs_own_err":  "deleg_auc",
        "deleg_auc":             "deleg_auc",
        "delegation_auc":        "deleg_auc",
        "Deleg_AUC":             "deleg_auc",
        "Deleg_AUC_80":          "deleg_auc",
        "model":                 "model",
        "Model":                 "model",
    }
    out = df.rename(columns=aliases)
    if "model" not in out.columns:
        raise ValueError("CSV missing model column (or alias)")
    needed = ["ece", "deleg_auc"]
    missing = [c for c in needed if c not in out.columns]
    if missing:
        raise ValueError(f"CSV missing required columns {missing}; got {list(df.columns)}")
    return out


def load_cohort(csv: Path, cohort_name: str, ece_kind: str) -> pd.DataFrame:
    df = pd.read_csv(csv)
    df = normalize_cols(df)
    df = df[["model", "ece", "deleg_auc"]].copy()
    df["cohort"] = cohort_name
    df["ece_kind"] = ece_kind
    return df


def render(df: pd.DataFrame, png: Path, pdf: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    plottable = df.dropna(subset=["ece", "deleg_auc"]).copy()
    excluded = df[df["deleg_auc"].isna() | df["ece"].isna()]
    print(f"Plottable: {len(plottable)} models")
    print(f"Excluded:  {len(excluded)} models  (Delegation AUC undefined: rate 0% or 100%)")

    style = {
        "Institutional": dict(marker="^", color="#1f77b4",
                              label="Institutional cohort (forced-answer ECE)", s=120),
        "Kaggle":        dict(marker="o", color="#d62728",
                              label="Kaggle cohort (answered-conditional ECE)", s=85),
    }

    for cohort, sub in plottable.groupby("cohort"):
        ax.scatter(sub["ece"], sub["deleg_auc"],
                   marker=style[cohort]["marker"],
                   c=style[cohort]["color"],
                   s=style[cohort]["s"],
                   edgecolor="black", linewidths=0.7,
                   alpha=0.85, label=style[cohort]["label"], zorder=3)

    # Annotate named examples from Sections 1 and 5.2
    highlights = {
        "llama3.2:1b":              ("llama3.2:1b\n(low ECE, near-chance routing)", ( 12, -22)),
        "llama3.2_1b":              ("llama3.2:1b\n(low ECE, near-chance routing)", ( 12, -22)),
        "llama3.1:8b":              ("llama3.1:8b\n(higher ECE, strong routing)",   (-145,   8)),
        "llama3.1_8b":              ("llama3.1:8b\n(higher ECE, strong routing)",   (-145,   8)),
        "claude-opus-4-6-default":  ("claude-opus-4-6-default\n(largest rank reversal)",
                                                                                    ( 12,  16)),
        "claude-haiku-4-5-20251001":("claude-haiku-4.5\n(top routing, mid ECE)",    ( 12, -22)),
    }
    for _, row in plottable.iterrows():
        m = str(row["model"])
        if m in highlights:
            label, (dx, dy) = highlights[m]
            ax.annotate(label, xy=(row["ece"], row["deleg_auc"]),
                        xytext=(dx, dy), textcoords="offset points",
                        fontsize=8, ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="0.3"))

    # Reference line at chance routing (label on the left side to avoid legend collision)
    ax.axhline(0.5, linestyle=":", color="0.5", linewidth=1, zorder=1)
    xlim = ax.get_xlim()
    ax.text(xlim[0] + 0.005, 0.505, "Delegation AUC = 0.5 (chance)", fontsize=8,
            color="0.4", ha="left", va="bottom")

    # Pad y-axis upper limit so the haiku annotation does not collide with title
    ax.set_ylim(top=ax.get_ylim()[1] + 0.04)

    ax.set_xlabel("ECE  (lower = better calibration)")
    ax.set_ylabel("Delegation AUC-ROC  (higher = better behavioral routing)")
    ax.set_title("Declarative calibration vs. behavioral routing")
    ax.grid(True, linestyle="-", linewidth=0.3, color="0.85", zorder=0)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaggle-csv", type=Path, required=True,
                    help="Per-model Kaggle metrics CSV (delegate_game_metrics.csv).")
    ap.add_argument("--institutional-csv", type=Path, default=None,
                    help="Optional per-model institutional comparative CSV with "
                         "ECE and Delegation AUC columns. If supplied, "
                         "institutional models are overlaid as a second series.")
    ap.add_argument("--output-png", type=Path, required=True)
    ap.add_argument("--output-pdf", type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    parts = [load_cohort(args.kaggle_csv, "Kaggle", "answered_conditional")]
    if args.institutional_csv is not None:
        if args.institutional_csv.exists():
            parts.append(load_cohort(args.institutional_csv, "Institutional", "forced_answer"))
        else:
            print(f"WARNING: --institutional-csv {args.institutional_csv} not found; "
                  f"plotting Kaggle cohort only", file=sys.stderr)
    df = pd.concat(parts, ignore_index=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved: {args.output_csv}")

    render(df, args.output_png, args.output_pdf)

    print()
    print("=== Per-cohort plottable counts ===")
    plottable = df.dropna(subset=["ece", "deleg_auc"])
    for c, sub in plottable.groupby("cohort"):
        print(f"  {c:20s}  n_plottable = {len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
