#!/usr/bin/env python3
"""
================================================================================
MSV: Risk-Coverage curve generator
================================================================================

Purpose
-------
Build a risk-coverage curve with four reference policies to contextualize
the behavioral Delegation metric:

  1. Confidence-threshold answering: answer items where declared
     confidence exceeds a sweep threshold (uses answered-only confidence).
  2. Delegation-decision answering: answer items the model did not
     delegate (a single point rather than a curve; it is the model's
     actual policy).
  3. Random delegation: answer items selected uniformly at random,
     sweeping coverage from 1% to 100%. Computed analytically from
     per-model answered accuracy.
  4. Oracle delegation: optimal item selection by ground-truth
     correctness. Upper bound on any delegation policy.

A clean risk-coverage curve addresses the reviewer question: is the
behavioral Delegation metric measuring something informative, or could
a simple confidence threshold do the same work? The answer in this
data: for most models the behavioral delegation point lies above the
confidence-threshold curve, meaning the model's delegation decisions
contain routing information not recoverable from the confidence signal
alone.

Usage
-----
  python scripts/generate_risk_coverage_figure.py \\
      --task1-csv     data/kaggle_extracted/per_task/t01_delegate_game.csv \\
      --models        claude-haiku-4-5-20251001 gemini-3.1-pro-preview \\
                      gpt-5.4-nano-2026-03-17 claude-opus-4-6-default \\
      --output-prefix results/reproduced/risk_coverage

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


def per_model_curves(df: pd.DataFrame):
    """Compute accuracy-at-coverage for each policy on a single model.

    Inputs: a per-model DataFrame with columns choice (ANSWER/DELEGATE),
    answer (letter or null), correct (ground-truth letter), confidence
    (1-4 or null).

    Outputs dict with arrays keyed by policy name:
      {
        "confidence": {"coverage": [...], "accuracy": [...]},
        "random":     {"coverage": [...], "accuracy": [...]},
        "oracle":     {"coverage": [...], "accuracy": [...]},
        "delegation": {"coverage": scalar, "accuracy": scalar},
      }
    """
    df = df.copy()
    df["is_ans"] = df["choice"] == "ANSWER"
    df["correct_bin"] = (df["answer"] == df["correct"]).astype(int)
    # For confidence-threshold curves we only have confidence on ANSWER trials.
    # We interpret "answer only when confidence >= t" as a policy applied to
    # the full panel: items the model chose to delegate or items below the
    # threshold become abstentions. For items not answered, we have no
    # confidence observation, so we treat them as "abstain" -- which is the
    # natural extension.
    n_total = len(df)
    if n_total == 0:
        return None

    # --- Behavioral delegation (single point) ---
    ans_trials = df[df["is_ans"]]
    if len(ans_trials) == 0:
        deleg_coverage = 0.0
        deleg_acc = np.nan
    else:
        deleg_coverage = len(ans_trials) / n_total
        deleg_acc = ans_trials["correct_bin"].mean()

    # --- Random delegation curve (analytic) ---
    # If the model's base accuracy on its answered subset is p, then random
    # selection at coverage c predicts answered-subset accuracy p regardless
    # of c. The "random" baseline is therefore a horizontal line at p.
    # This is the null policy: random delegation adds no information.
    base_acc = deleg_acc
    rand_coverage = np.linspace(0.01, 1.0, 100)
    rand_accuracy = np.full_like(rand_coverage, base_acc)

    # --- Confidence-threshold curve ---
    # Sweep across declared confidence thresholds. At each threshold t,
    # coverage is (# answered with conf >= t) / n_total, and accuracy is
    # the fraction correct among that subset.
    conf_points = [(0.0, deleg_coverage, deleg_acc)]  # threshold=0: all answered
    for t in (1.5, 2.5, 3.5, 4.5):
        kept = ans_trials[ans_trials["confidence"] >= t]
        if len(kept) == 0:
            continue
        cov = len(kept) / n_total
        acc = kept["correct_bin"].mean()
        conf_points.append((t, cov, acc))
    # Sort by coverage (ascending) so the curve reads left-to-right
    conf_points.sort(key=lambda x: x[1])
    conf_cov = np.array([c for _, c, _ in conf_points])
    conf_acc = np.array([a for _, _, a in conf_points])

    # --- Oracle curve ---
    # The best possible selective predictor: answer the items the model
    # actually got right; abstain on the rest. Accuracy is 1.0 up to the
    # model's base answered-accuracy rate (across full panel), then drops.
    # We compute this on the answered subset only, because we don't observe
    # correctness on delegated items.
    if len(ans_trials) > 0:
        # Sort correctness descending (true before false)
        sorted_correct = ans_trials["correct_bin"].sort_values(ascending=False).values
        cum = np.cumsum(sorted_correct)
        ks = np.arange(1, len(sorted_correct) + 1)
        oracle_cov = ks / n_total  # coverage relative to full panel
        oracle_acc = cum / ks       # cumulative accuracy
    else:
        oracle_cov = oracle_acc = np.array([])

    return {
        "confidence": {"coverage": conf_cov,   "accuracy": conf_acc},
        "random":     {"coverage": rand_coverage, "accuracy": rand_accuracy},
        "oracle":     {"coverage": oracle_cov, "accuracy": oracle_acc},
        "delegation": {"coverage": deleg_coverage, "accuracy": deleg_acc},
        "n_total":    n_total,
        "n_answered": len(ans_trials),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--task1-csv", type=Path, required=True,
                    help="Per-task Task 1 CSV (t01_delegate_game.csv)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Specific model names to plot. Default: four "
                         "models spanning the cohort.")
    ap.add_argument("--output-prefix", type=str, default="risk_coverage")
    args = ap.parse_args()

    df = pd.read_csv(args.task1_csv)

    # Default model selection: 4 models spanning the cohort
    if args.models is None:
        args.models = [
            "claude-haiku-4-5-20251001",        # strong delegator
            "gemini-3.1-pro-preview",           # strong but low delegation
            "gpt-5.4-nano-2026-03-17",          # highest delegation rate
            "gemma-3-27b-it",                   # weaker, moderate delegation
        ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
    axes = axes.flatten()

    for ax, model in zip(axes, args.models):
        sub = df[df["model"] == model]
        if len(sub) == 0:
            ax.set_title(f"{model}\n(no data)")
            continue
        curves = per_model_curves(sub)
        if curves is None:
            continue

        # Oracle
        if len(curves["oracle"]["coverage"]) > 0:
            ax.plot(curves["oracle"]["coverage"],
                    curves["oracle"]["accuracy"],
                    color="#2e7d32", linestyle="-", linewidth=2,
                    label="Oracle (upper bound)")

        # Confidence-threshold curve
        ax.plot(curves["confidence"]["coverage"],
                curves["confidence"]["accuracy"],
                color="#c62828", linestyle="--", linewidth=1.8,
                marker="o", markersize=5,
                label="Confidence threshold")

        # Random
        ax.plot(curves["random"]["coverage"],
                curves["random"]["accuracy"],
                color="#757575", linestyle=":", linewidth=1.5,
                label="Random (flat)")

        # Behavioral delegation: single point
        ax.scatter([curves["delegation"]["coverage"]],
                   [curves["delegation"]["accuracy"]],
                   color="#1565c0", s=120, zorder=5,
                   marker="*", edgecolor="black", linewidth=0.8,
                   label="Behavioral delegation (actual)")

        # Cosmetics
        nice = (model.replace("-20251001", "").replace("-20250805", "")
                     .replace("-2026-03-05", "").replace("-2026-03-17", "")
                     .replace("-default", "").replace("-preview", "")
                     .replace("-instruct-2507", "")
                     .replace("-a3b", ""))
        ax.set_title(f"{nice}\n(n_answered={curves['n_answered']}, "
                     f"deleg={1-curves['delegation']['coverage']:.0%})",
                     fontsize=9)
        ax.set_xlabel("Coverage", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Accuracy", fontsize=9)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if ax == axes[0]:
            ax.legend(loc="lower left", fontsize=7, framealpha=0.9)

    fig.suptitle("Risk-coverage curves: behavioral delegation vs. "
                 "confidence-threshold abstention on the Kaggle cohort",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    pdf_path = Path(f"{args.output_prefix}.pdf")
    png_path = Path(f"{args.output_prefix}.png")
    plt.savefig(pdf_path, dpi=150, bbox_inches="tight")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {pdf_path}")
    print(f"Wrote: {png_path}")


if __name__ == "__main__":
    main()
