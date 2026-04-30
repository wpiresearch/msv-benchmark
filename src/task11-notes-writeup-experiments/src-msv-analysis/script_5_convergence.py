"""
Script 5: Cross-task convergence matrix and heatmap.

Integrates the outputs of Scripts 2, 3, and 4 into a single summary matrix
with one row per model, showing how metacognitive signals converge across
tasks. This is the headline figure for the NeurIPS paper: a clean separation
of reasoning models from non-reasoning models across three independent
signals produces the most persuasive evidence for the metacognitive
inefficiency finding.

Columns in the matrix:
  - is_reasoning             (0/1, from catalog)
  - t01_delegate_slope       (from Script 3 slopes CSV; negative or near
                             zero for reasoning models, positive for
                             non-reasoning models)
  - t01_delegate_rate        (overall delegation rate on Task 1)
  - t02_coherence_corr       (from Script 4: declared activation vs routing
                             action Spearman rho)
  - t11_d_hat                (from Script 2: object-level discrimination)
  - t11_type2_auc            (from Script 2: metacognitive discrimination)
  - t11_mc                   (from Script 2: efficiency ratio)
  - verbosity_index          (from Script 1: mean output tokens/response)

The heatmap uses per-column z-score normalization so signals on different
scales can be compared visually. A "metacognitive inefficiency score" column
is also computed as a weighted combination of the three signals pointing in
the same direction, with reasoning models expected to cluster at one end.

Inputs:
  --catalog:              outputs/run_catalog.csv
  --verbosity-stats:      outputs/verbosity_stats.csv  (Script 1)
  --task11-stats:         outputs/task11_metacognitive_efficiency.csv  (Script 2)
  --task1-slopes:         outputs/task1_delegation_curves_slopes.csv  (Script 3)
  --task2-coherence:      outputs/task2_coherence.csv  (Script 4)
  --out-csv:              outputs/convergence_matrix.csv
  --out-fig:              outputs/convergence_heatmap.png

Outputs:
  convergence_matrix.csv: one row per model with all signals joined.
  convergence_heatmap.png: heatmap of z-scored signals with models ordered
    by is_reasoning (reasoning models at top), and a dividing line between
    the two groups.

Usage:
  python script_5_convergence.py \\
      --catalog outputs/run_catalog.csv \\
      --verbosity-stats outputs/verbosity_stats.csv \\
      --task11-stats outputs/task11_metacognitive_efficiency.csv \\
      --task1-slopes outputs/task1_delegation_curves_slopes.csv \\
      --task2-coherence outputs/task2_coherence.csv \\
      --out-csv outputs/convergence_matrix.csv \\
      --out-fig outputs/convergence_heatmap.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    ("verbosity_index", False),          # higher = more reasoning-like
    ("t01_delegate_rate", True),         # higher = more metacog functional
    ("t01_delegate_slope", True),        # higher = more metacog functional
    ("t02_coherence_corr", True),        # higher = more metacog functional
    ("t11_d_hat", True),                 # higher = better object-level ability
    ("t11_type2_auc", True),             # higher = better metacognition
    ("t11_mc", True),                    # higher = better efficiency
]


def load_and_merge(catalog_path: Path, verbosity_path: Path,
                   task11_path: Path, task1_slopes_path: Path,
                   task2_path: Path) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    models = catalog[["model", "is_reasoning"]].drop_duplicates()

    # Verbosity: average verbosity_index across tasks per model
    if verbosity_path.exists():
        vb = pd.read_csv(verbosity_path)
        vb_agg = vb.groupby("model")["verbosity_index"].mean().reset_index()
    else:
        vb_agg = pd.DataFrame(columns=["model", "verbosity_index"])

    # Task 11 efficiency
    if task11_path.exists():
        t11 = pd.read_csv(task11_path)
        t11 = t11[["model", "d_hat", "type2_auc", "mc"]].rename(columns={
            "d_hat": "t11_d_hat",
            "type2_auc": "t11_type2_auc",
            "mc": "t11_mc",
        })
    else:
        t11 = pd.DataFrame(columns=["model", "t11_d_hat", "t11_type2_auc", "t11_mc"])

    # Task 1 slopes
    if task1_slopes_path.exists():
        t1 = pd.read_csv(task1_slopes_path)
        t1 = t1[["model", "delegation_slope", "overall_delegate_rate"]].rename(columns={
            "delegation_slope": "t01_delegate_slope",
            "overall_delegate_rate": "t01_delegate_rate",
        })
    else:
        t1 = pd.DataFrame(columns=["model", "t01_delegate_slope", "t01_delegate_rate"])

    # Task 2 coherence
    if task2_path.exists():
        t2 = pd.read_csv(task2_path)
        t2 = t2[["model", "activation_routing_corr"]].rename(columns={
            "activation_routing_corr": "t02_coherence_corr",
        })
    else:
        t2 = pd.DataFrame(columns=["model", "t02_coherence_corr"])

    merged = (models.merge(vb_agg, on="model", how="left")
                    .merge(t1, on="model", how="left")
                    .merge(t2, on="model", how="left")
                    .merge(t11, on="model", how="left"))
    return merged


def compute_inefficiency_score(df: pd.DataFrame) -> pd.Series:
    """
    Single scalar summary: higher = more metacognitive inefficiency.
    Combines three independently-measurable signals that each point to
    reasoning-model-like behavior (high verbosity, low delegation slope,
    low metacognitive efficiency).
    """
    score = pd.Series(0.0, index=df.index, dtype=float)
    n_signals = 0

    # Signal 1: high verbosity
    if "verbosity_index" in df and df["verbosity_index"].notna().any():
        v = df["verbosity_index"]
        v_norm = (v - v.min()) / (v.max() - v.min() + 1e-9)
        score += v_norm.fillna(0.0)
        n_signals += 1

    # Signal 2: low delegation slope
    if "t01_delegate_slope" in df and df["t01_delegate_slope"].notna().any():
        s = df["t01_delegate_slope"]
        s_norm = 1.0 - (s - s.min()) / (s.max() - s.min() + 1e-9)
        score += s_norm.fillna(0.0)
        n_signals += 1

    # Signal 3: low metacognitive efficiency
    if "t11_mc" in df and df["t11_mc"].notna().any():
        m = df["t11_mc"]
        m_norm = 1.0 - (m - m.min()) / (m.max() - m.min() + 1e-9)
        score += m_norm.fillna(0.0)
        n_signals += 1

    if n_signals == 0:
        return score
    return (score / n_signals).round(4)


def zscore(series: pd.Series) -> pd.Series:
    mu = series.mean(skipna=True)
    sd = series.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sd


def build_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    """Z-scored heatmap, reasoning models at top."""
    df = df.copy().set_index("model")
    # Order: reasoning first (higher inefficiency_score), then others
    df = df.sort_values(["is_reasoning", "inefficiency_score"],
                        ascending=[False, False])

    signal_cols = [c for c, _ in SIGNAL_COLUMNS if c in df.columns]
    z = df[signal_cols].apply(zscore)

    fig, ax = plt.subplots(figsize=(max(8, len(signal_cols) * 1.4),
                                    max(4, len(df) * 0.4)))
    im = ax.imshow(z.values, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)

    ax.set_xticks(np.arange(len(signal_cols)))
    ax.set_xticklabels(signal_cols, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df.index, fontsize=9)

    # Separator line between reasoning and non-reasoning
    n_reasoning = int(df["is_reasoning"].sum())
    if 0 < n_reasoning < len(df):
        ax.axhline(n_reasoning - 0.5, color="black", lw=1.5)

    # Annotate each cell with the original (non-z-scored) value
    for i, model in enumerate(df.index):
        for j, col in enumerate(signal_cols):
            val = df.loc[model, col]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    cbar = plt.colorbar(im, ax=ax, shrink=0.75)
    cbar.set_label("z-score across models", fontsize=9)
    ax.set_title("Cross-task metacognitive signal convergence\n"
                 "Reasoning-enhanced models (above line) vs non-reasoning (below)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--verbosity-stats", type=Path, required=True)
    ap.add_argument("--task11-stats", type=Path, required=True)
    ap.add_argument("--task1-slopes", type=Path, required=True)
    ap.add_argument("--task2-coherence", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    df = load_and_merge(args.catalog, args.verbosity_stats, args.task11_stats,
                        args.task1_slopes, args.task2_coherence)
    df["inefficiency_score"] = compute_inefficiency_score(df)
    df = df.sort_values(["is_reasoning", "inefficiency_score"],
                        ascending=[False, False])

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("Convergence matrix:")
    print(df.to_string(index=False))
    build_heatmap(df, args.out_fig)
    print(f"\nMatrix CSV: {args.out_csv}")
    print(f"Heatmap:    {args.out_fig}")


if __name__ == "__main__":
    main()
