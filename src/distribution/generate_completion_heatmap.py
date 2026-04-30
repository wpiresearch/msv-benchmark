#!/usr/bin/env python3
"""
================================================================================
MSV: Completion x Task heatmap
================================================================================

Purpose
-------
Generates a 23-model x 11-task heatmap showing the completion outcome for
each (model, task) run on the Kaggle cohort. Completion outcomes:
  - clean      : run completed without budget/quota/other failures
  - budget     : hit Kaggle platform budget or quota limit
  - other      : other platform error (API 503, 400, timeout, context)

This figure turns the Kaggle completion-heterogeneity concern (ChatGPT
Pro flagged this as a central E&D finding) into an explicit benchmark-
execution artifact. The reviewer's point is that benchmark execution
itself is part of the evaluation object; the heatmap makes it visible.

Usage
-----
  python scripts/generate_completion_heatmap.py \\
      --metadata-csv  data/kaggle_extracted/run_metadata.csv \\
      --output-prefix results/reproduced/completion_heatmap

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
from matplotlib.colors import ListedColormap, BoundaryNorm


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--metadata-csv", type=Path, required=True,
                    help="run_metadata.csv from extract_kaggle_outputs.py")
    ap.add_argument("--output-prefix", type=str, default="completion_heatmap")
    args = ap.parse_args()

    md = pd.read_csv(args.metadata_csv)
    # Classify each run
    def classify(r):
        if r["budget_failure"] == True or str(r["budget_failure"]).lower() == "true":
            return 1  # budget
        if isinstance(r.get("other_failure"), str) and r["other_failure"].strip():
            return 2  # other
        return 0  # clean

    md["outcome"] = md.apply(classify, axis=1)

    # Pivot to model x task
    pivot = md.pivot_table(
        index="model", columns="task_id", values="outcome",
        aggfunc="first",
    )
    # Order tasks t01..t11
    task_order = [f"t{i:02d}" for i in range(1, 12)]
    pivot = pivot.reindex(columns=task_order)
    # Order models by total clean count (most clean at top)
    clean_count = (pivot == 0).sum(axis=1).sort_values(ascending=False)
    pivot = pivot.reindex(index=clean_count.index)

    # Display name fixups
    display_models = (pivot.index
                      .str.replace("-20251001",   "")
                      .str.replace("-20250805",   "")
                      .str.replace("-2026-03-05", "")
                      .str.replace("-2026-03-17", "")
                      .str.replace("-default",    "")
                      .str.replace("-preview",    "")
                      .str.replace("-instruct-2507", "")
                      .str.replace("-a3b-instruct", "-inst")
                      .str.replace("-a3b-thinking", "-think")
                      .str.replace("-a35b-instruct","-coder"))
    display_tasks = [
        "T1\nDelegate", "T2\nDeclared", "T3\nRevise",
        "T4\nConfEnt",  "T5\nTeammate", "T6\nER",
        "T7\nCI",       "T8\nEM",       "T9\nPI",
        "T10\nDPP",     "T11\nMC",
    ]

    # Color map: 0=green (clean), 1=orange (budget), 2=red (other)
    cmap = ListedColormap(["#2e7d32", "#f57c00", "#c62828", "#f5f5f5"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    # NaN cells (no run recorded) get outcome = 3 -> grey
    mat = pivot.fillna(3).values.astype(float)
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels(display_tasks, fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(display_models, fontsize=8)

    # Completion summary in the title
    total_runs = int((pivot.notna()).values.sum())
    n_clean   = int((pivot == 0).values.sum())
    n_budget  = int((pivot == 1).values.sum())
    n_other   = int((pivot == 2).values.sum())
    ax.set_title(
        f"Kaggle benchmark completion picture: 23 models x 11 tasks = "
        f"{total_runs} runs\n"
        f"{n_clean} clean ({n_clean/total_runs:.0%}), "
        f"{n_budget} budget/quota ({n_budget/total_runs:.0%}), "
        f"{n_other} other errors ({n_other/total_runs:.0%})",
        fontsize=10.5, pad=12,
    )

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(task_order)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index)), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Legend (below the plot)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2e7d32", edgecolor="black", label="Clean"),
        Patch(facecolor="#f57c00", edgecolor="black", label="Budget / quota limit"),
        Patch(facecolor="#c62828", edgecolor="black", label="Other platform error"),
    ]
    ax.legend(handles=legend_elements, loc="upper center",
              bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=9, frameon=False)

    plt.tight_layout()
    pdf_path = Path(f"{args.output_prefix}.pdf")
    png_path = Path(f"{args.output_prefix}.png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")

    # Per-task summary CSV
    task_stats = []
    for t in task_order:
        col = pivot[t]
        task_stats.append({
            "task":        t,
            "n_runs":      int(col.notna().sum()),
            "n_clean":     int((col == 0).sum()),
            "n_budget":    int((col == 1).sum()),
            "n_other":     int((col == 2).sum()),
        })
    pd.DataFrame(task_stats).to_csv(
        f"{args.output_prefix}_per_task.csv", index=False,
    )
    print(f"Wrote: {args.output_prefix}_per_task.csv")


if __name__ == "__main__":
    main()
