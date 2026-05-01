"""
Script 6: Verbosity-to-metacognitive-inefficiency correlation.

Tests whether chain-of-thought verbosity (mean output tokens per response)
correlates with metacognitive inefficiency (MC = d*/d_hat) across models.
Two possible outcomes:

  - If verbosity and inefficiency are strongly correlated (negative
    Pearson/Spearman between verbosity and MC), the two phenomena
    likely share a common cause in reasoning-focused training.
  - If they dissociate (some verbose models have non-zero MC, or some
    terse models have zero MC), they are separable phenomena that must
    be discussed separately in the paper.

The script also computes the correlation for two other signals from the
convergence matrix (delegation slope and declared-routing coherence), in
case the story is cleaner in those dimensions than in MC.

Inputs:
  --convergence: outputs/convergence_matrix.csv from Script 5
  --out-csv:     outputs/verbosity_vs_efficiency.csv
  --out-fig:     outputs/verbosity_vs_efficiency.png

Outputs:
  verbosity_vs_efficiency.csv: correlation coefficients (Pearson and
    Spearman) between verbosity_index and each of the three metacognitive
    signals (t11_mc, t01_delegate_slope, t02_coherence_corr).

  verbosity_vs_efficiency.png: three-panel scatter plot with verbosity on
    the x-axis and each of the three metacognitive signals on the y-axis.
    Each panel shows Pearson and Spearman correlations in its title.
    Points colored by reasoning/non-reasoning category.

Usage:
  python script_6_verbosity_vs_efficiency.py \\
      --convergence outputs/convergence_matrix.csv \\
      --out-csv outputs/verbosity_vs_efficiency.csv \\
      --out-fig outputs/verbosity_vs_efficiency.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


SIGNALS = [
    ("t11_mc", "MC (metacognitive efficiency)"),
    ("t01_delegate_slope", "Task 1 delegation slope"),
    ("t02_coherence_corr", "Task 2 declared-routing coherence"),
]


def safe_corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson and Spearman; returns NaN if insufficient data."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 3:
        return {"pearson": np.nan, "pearson_p": np.nan,
                "spearman": np.nan, "spearman_p": np.nan}
    x_m, y_m = x[mask], y[mask]
    if np.std(x_m) < 1e-9 or np.std(y_m) < 1e-9:
        return {"pearson": np.nan, "pearson_p": np.nan,
                "spearman": np.nan, "spearman_p": np.nan}
    p_r, p_p = pearsonr(x_m, y_m)
    s_r, s_p = spearmanr(x_m, y_m)
    return {"pearson": round(float(p_r), 4),
            "pearson_p": round(float(p_p), 4),
            "spearman": round(float(s_r), 4),
            "spearman_p": round(float(s_p), 4)}


def build_scatter(df: pd.DataFrame, corrs: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(SIGNALS), figsize=(5 * len(SIGNALS), 5))
    if len(SIGNALS) == 1:
        axes = [axes]

    reas_mask = df["is_reasoning"].astype(bool).values
    x = df["verbosity_index"].values

    for ax, (col, label) in zip(axes, SIGNALS):
        y = df[col].values
        # Non-reasoning first so reasoning points appear on top
        ax.scatter(x[~reas_mask], y[~reas_mask],
                   color="#4c72b0", alpha=0.85, s=60, label="non-reasoning")
        ax.scatter(x[reas_mask], y[reas_mask],
                   color="#c44e52", alpha=0.85, s=60, label="reasoning")

        # Annotate each point with model name
        for i in range(len(df)):
            if np.isnan(x[i]) or np.isnan(y[i]):
                continue
            ax.annotate(df["model"].iloc[i], (x[i], y[i]),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)

        ax.set_xlabel("Verbosity index (mean output tokens)")
        ax.set_ylabel(label)
        c = corrs.get(col, {})
        title = f"{label}\n"
        if not np.isnan(c.get("pearson", np.nan)):
            title += (f"Pearson r = {c['pearson']:+.3f} (p={c['pearson_p']:.3f}), "
                      f"Spearman ρ = {c['spearman']:+.3f} (p={c['spearman_p']:.3f})")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--convergence", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.convergence)
    if "verbosity_index" not in df.columns:
        print("ERROR: convergence_matrix.csv must include verbosity_index.")
        return

    x = df["verbosity_index"].values.astype(float)
    corrs = {}
    rows = []
    for col, label in SIGNALS:
        if col not in df.columns:
            continue
        y = df[col].values.astype(float)
        c = safe_corr(x, y)
        corrs[col] = c
        rows.append({
            "signal": col,
            "signal_label": label,
            "n_models": int((~(np.isnan(x) | np.isnan(y))).sum()),
            **c,
        })

    if not rows:
        print("No signals available for correlation analysis.")
        return

    out_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print("Verbosity vs metacognitive signal correlations:")
    print(out_df.to_string(index=False))

    build_scatter(df, corrs, args.out_fig)
    print(f"\nCorrelations CSV: {args.out_csv}")
    print(f"Figure:           {args.out_fig}")


if __name__ == "__main__":
    main()
