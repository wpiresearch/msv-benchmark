"""
Script 4: Task 2 declared-vs-behavioral coherence analysis.

Task 2 (Declared MSV Probe) asks each model to:
  1. Rate its MSV (Metacognitive State Vector) on five dimensions (CE, ER,
     CI, EM, PI) for each question.
  2. Declare a routing action: ANSWER, DELIBERATE, or DELEGATE.

If a model's declared MSV values correlate with its routing action in the
expected direction (high declared activation -> DELIBERATE or DELEGATE),
the model's self-report has behavioral teeth -- it translates into action.
If declared activation is high but routing is uniformly ANSWER, the model
produces structured self-reports that do not govern its behavior. This is
the declared-vs-behavioral mismatch hypothesis (H3 in the testing plan).

Expected CSV columns for Task 2:
  - question_id
  - declared_activation: float, a scalar summary of the five MSV values
    (usually the equal-weight mean of CE, ER, CI, EM, PI; adjust
    ACTIVATION_COL below if your CSV uses a different name)
  - routing_action: "ANSWER" | "DELIBERATE" | "DELEGATE"
  - Optionally: individual declared MSV values (CE, ER, CI, EM, PI)

Inputs:
  --catalog: path to run_catalog.csv
  --task-id: default "t02"
  --out-csv: path to write task2_coherence.csv
  --out-fig: path to write task2_coherence_scatter.png

Outputs:
  task2_coherence.csv: one row per model with:
    n_questions, activation_mean, activation_std, answer_rate,
    deliberate_rate, delegate_rate, activation_routing_corr (Spearman rho
    between declared activation and ordered routing action)

  task2_coherence_scatter.png: small-multiples plot, one subplot per
    model, scatter of declared_activation (x) vs routing_action (y,
    coded ANSWER=0, DELIBERATE=1, DELEGATE=2). Reasoning models should
    show flat horizontal scatter at y=0; non-reasoning models should
    show positive slope.

Usage:
  python script_4_task2_coherence.py \\
      --catalog outputs/run_catalog.csv \\
      --out-csv outputs/task2_coherence.csv \\
      --out-fig outputs/task2_coherence_scatter.png
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Adjust if your Task 2 CSV uses different names
ACTIVATION_COL = "declared_activation"
ROUTING_COL = "routing_choice"  # actual column in t02 CSVs
MSV_DIMENSION_COLS = ["CE", "ER", "CI", "EM", "PI"]  # fallback source
ACTION_CODE = {"ANSWER": 0, "DELIBERATE": 1, "DELEGATE": 2}


def coerce_activation(df: pd.DataFrame) -> pd.Series:
    """Return a declared-activation series, computing it from MSV dims if needed."""
    if ACTIVATION_COL in df.columns:
        return df[ACTIVATION_COL].astype(float)
    dims_present = [c for c in MSV_DIMENSION_COLS if c in df.columns]
    if len(dims_present) >= 3:
        return df[dims_present].astype(float).mean(axis=1)
    raise KeyError(
        f"Could not find '{ACTIVATION_COL}' or enough MSV dimension columns "
        f"({MSV_DIMENSION_COLS}) in Task 2 CSV"
    )


def compute_coherence(df: pd.DataFrame) -> dict:
    """Summarize one model's declared-vs-behavioral coherence."""
    if ROUTING_COL not in df.columns:
        return {}
    df = df.copy()
    df["_action_int"] = df[ROUTING_COL].astype(str).str.upper().map(ACTION_CODE)
    df = df.dropna(subset=["_action_int"])
    df["_activation"] = coerce_activation(df)
    df = df.dropna(subset=["_activation"])

    if len(df) < 5:
        return {}

    n = len(df)
    activation = df["_activation"].values
    actions = df["_action_int"].values

    # Spearman because routing action is ordered categorical
    if np.std(actions) < 1e-9 or np.std(activation) < 1e-9:
        corr, pval = 0.0, 1.0
    else:
        corr, pval = spearmanr(activation, actions)
        corr = float(corr) if not np.isnan(corr) else 0.0

    counts = df[ROUTING_COL].astype(str).str.upper().value_counts()
    return {
        "n_questions": n,
        "activation_mean": round(float(activation.mean()), 4),
        "activation_std": round(float(activation.std()), 4),
        "answer_rate": round(counts.get("ANSWER", 0) / n, 4),
        "deliberate_rate": round(counts.get("DELIBERATE", 0) / n, 4),
        "delegate_rate": round(counts.get("DELEGATE", 0) / n, 4),
        "activation_routing_corr": round(corr, 4),
        "corr_pvalue": round(float(pval), 4),
    }


def build_coherence_scatter(per_model_data: dict, out_path: Path) -> None:
    """Small-multiples: one subplot per model."""
    models = list(per_model_data.keys())
    n_models = len(models)
    if n_models == 0:
        return
    n_cols = min(3, n_models)
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.2 * n_rows),
                             squeeze=False)
    for i, model in enumerate(models):
        ax = axes[i // n_cols][i % n_cols]
        data = per_model_data[model]
        if data is None or len(data["activation"]) == 0:
            ax.set_visible(False)
            continue

        # Jitter the y-coordinate slightly so overlapping points are visible
        rng = np.random.default_rng(42)
        y_jitter = data["actions"] + rng.uniform(-0.15, 0.15, len(data["actions"]))
        color = "#c44e52" if data["is_reasoning"] else "#4c72b0"
        ax.scatter(data["activation"], y_jitter,
                   alpha=0.6, color=color, s=25)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["ANSWER", "DELIBERATE", "DELEGATE"], fontsize=8)
        ax.set_xlabel("Declared activation", fontsize=9)
        ax.set_title(f"{model}\n(rho = {data['corr']:+.3f})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.5, 2.5)

    # Hide unused subplots
    for j in range(i + 1, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].set_visible(False)

    fig.suptitle("Declared MSV activation vs chosen routing action (Task 2)\n"
                 "Reasoning models (red) should show flat scatter at ANSWER",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--task-id", default="t02")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    catalog = pd.read_csv(args.catalog)
    task_rows = catalog[catalog["task_id"] == args.task_id]

    summaries = []
    per_model_data = {}

    for _, cat_row in task_rows.iterrows():
        csv_path = cat_row.get("csv_file_path")
        if not isinstance(csv_path, str) or not Path(csv_path).exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read {cat_row['model']}: {e}")
            continue

        try:
            coh = compute_coherence(df)
        except KeyError as e:
            print(f"Skipping {cat_row['model']}: {e}")
            continue

        if not coh:
            continue

        summaries.append({
            "model": cat_row["model"],
            "is_reasoning": bool(cat_row["is_reasoning"]),
            **coh,
        })

        df_use = df.copy()
        df_use["_action_int"] = df_use[ROUTING_COL].astype(str).str.upper().map(ACTION_CODE)
        df_use = df_use.dropna(subset=["_action_int"])
        try:
            df_use["_activation"] = coerce_activation(df_use)
        except KeyError:
            continue
        df_use = df_use.dropna(subset=["_activation"])

        per_model_data[cat_row["model"]] = {
            "activation": df_use["_activation"].values,
            "actions": df_use["_action_int"].values,
            "is_reasoning": bool(cat_row["is_reasoning"]),
            "corr": coh["activation_routing_corr"],
        }

    if not summaries:
        print(f"No {args.task_id} data could be analyzed.")
        return

    out_df = pd.DataFrame(summaries).sort_values(
        "activation_routing_corr", ascending=True
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Analyzed {len(out_df)} models on {args.task_id}.")
    print()
    print(out_df.to_string(index=False))

    build_coherence_scatter(per_model_data, args.out_fig)
    print(f"\nCoherence CSV: {args.out_csv}")
    print(f"Figure:        {args.out_fig}")


if __name__ == "__main__":
    main()
