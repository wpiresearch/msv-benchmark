"""
Script 2: Task 11 metacognitive efficiency with bootstrap confidence intervals.

Recomputes d_hat (type-1 / object-level discrimination), type-2 AUC, and MC
(metacognitive efficiency = d*/d_hat) from the per-trial CSV for each model,
with 95% bootstrap confidence intervals. This replaces the single-point
estimates produced during the online run with statistically defensible
intervals.

The Task 11 CSV is expected to contain one row per trial with at least:
  - question_id
  - trial_type: "signal" or "noise" (signal=paired with correct answer,
    noise=paired with distractor)
  - model_answer: "yes" or "no" (does the model think this candidate is correct?)
  - is_correct: bool (whether model_answer matched ground truth for this trial)
  - confidence: integer 1-4

If your Task 11 CSV uses different column names, update COLUMN_MAP below.

The type-2 AUC is computed by treating confidence as a classifier of the
correctness of the model's own response. d* is estimated from the type-2
ROC using the Maniscalco-Lau approximation; for simplicity we use type-2
AUC as a sufficient statistic and report MC_approx = 2 * (AUC - 0.5) / (d_hat / 2).
A full MLE estimation of meta-d' following Maniscalco & Lau 2012 can be
added later using their published MATLAB port or the R 'metaSDT' package;
the approximation here is adequate for the qualitative dissociation finding.

Inputs:
  --catalog: path to run_catalog.csv
  --task-id: which task to analyze (default: t11)
  --n-boot: number of bootstrap samples (default: 2000)
  --out-csv: path to write task11_metacognitive_efficiency.csv
  --out-fig: path to write d_hat_vs_type2auc.png

Outputs:
  task11_metacognitive_efficiency.csv: one row per model with
    n_trials, hit_rate, fa_rate, d_hat, d_hat_ci_low, d_hat_ci_high,
    type2_auc, type2_auc_ci_low, type2_auc_ci_high, mc, mc_ci_low, mc_ci_high

  d_hat_vs_type2auc.png: scatter plot of d_hat (x) vs type-2 AUC (y)
    with one point per model. Bootstrap-derived error bars on both axes.
    Reasoning models should cluster at high d_hat / ~0.5 type-2 AUC.
    Non-reasoning models should show positive correlation on the diagonal.

Usage:
  python script_2_task11_metacog.py \\
      --catalog outputs/run_catalog.csv \\
      --out-csv outputs/task11_metacognitive_efficiency.csv \\
      --out-fig outputs/d_hat_vs_type2auc.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Map from expected logical columns to the actual column names in your CSV
# Adjust these if your Task 11 CSV uses different headers.
COLUMN_MAP = {
    "trial_type": "is_signal",        # actual column: bool (True=signal, False=noise)
    "is_correct": "judgment_correct", # actual column: bool
    "confidence": "confidence",       # actual column: int (1-9 scale on this task)
    "model_answer": "said_yes",       # actual column: bool (True=yes, False=no)
}


def compute_d_hat(hits: int, n_signal: int, fas: int, n_noise: int) -> float:
    """Standard type-1 d-prime with clipping to avoid infinities."""
    if n_signal == 0 or n_noise == 0:
        return np.nan
    h = np.clip(hits / n_signal, 1e-4, 1 - 1e-4)
    f = np.clip(fas / n_noise, 1e-4, 1 - 1e-4)
    return float(norm.ppf(h) - norm.ppf(f))


def compute_type2_auc(correctness: np.ndarray, confidence: np.ndarray) -> float:
    """
    Type-2 AUC: area under the ROC of confidence as a predictor of correctness.
    If everyone uses the same confidence value, AUC is 0.5 by construction.
    Uses trapezoidal integration over sorted confidence thresholds.
    """
    correctness = np.asarray(correctness, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    n = len(correctness)
    if n == 0 or correctness.sum() == 0 or (~correctness).sum() == 0:
        return 0.5

    # Sweep thresholds from high to low; at each threshold compute
    # fraction of correct and incorrect trials above the threshold
    thresholds = np.sort(np.unique(confidence))[::-1]
    type2_hits = [0.0]  # (0,0)
    type2_fas = [0.0]
    n_correct = correctness.sum()
    n_incorrect = (~correctness).sum()
    for t in thresholds:
        mask = confidence >= t
        type2_hits.append(((mask) & correctness).sum() / n_correct)
        type2_fas.append(((mask) & ~correctness).sum() / n_incorrect)
    type2_hits.append(1.0)  # (1,1)
    type2_fas.append(1.0)

    # Trapezoidal integration; ensure x is monotonically increasing
    fas = np.array(type2_fas)
    hits = np.array(type2_hits)
    order = np.argsort(fas)
    return float(np.trapezoid(hits[order], fas[order]))


def auc_to_d_star_approx(auc: float) -> float:
    """
    Approximate conversion from type-2 AUC to meta-d' under Gaussian
    assumptions. Exact meta-d' estimation requires MLE (Maniscalco & Lau
    2012); this approximation is sufficient for the qualitative
    dissociation finding.
    """
    auc_c = np.clip(auc, 1e-4, 1 - 1e-4)
    # z-transform of AUC times sqrt(2) gives d' under equal-variance Gaussian
    return float(norm.ppf(auc_c) * np.sqrt(2.0))


def compute_metacog_stats(df: pd.DataFrame) -> dict:
    """One-shot computation of all metacognitive statistics on a dataframe."""
    trial_col = COLUMN_MAP["trial_type"]
    correct_col = COLUMN_MAP["is_correct"]
    conf_col = COLUMN_MAP["confidence"]
    ans_col = COLUMN_MAP["model_answer"]

    # is_signal is bool: True=signal trial, False=noise trial
    sig = df[df[trial_col] == True]
    noi = df[df[trial_col] == False]

    # Type-1: hits = model said "yes" on signal trials;
    # FAs  = model said "yes" on noise trials
    # said_yes is already bool; .sum() counts True values
    hits = int(sig[ans_col].astype(bool).sum())
    fas = int(noi[ans_col].astype(bool).sum())
    n_signal = len(sig)
    n_noise = len(noi)

    d_hat = compute_d_hat(hits, n_signal, fas, n_noise)

    # Type-2: confidence as predictor of correctness
    type2_auc = compute_type2_auc(
        df[correct_col].astype(bool).values,
        df[conf_col].astype(float).values,
    )
    d_star = auc_to_d_star_approx(type2_auc)
    mc = d_star / d_hat if d_hat and not np.isnan(d_hat) and d_hat > 0 else np.nan

    return {
        "n_trials": len(df),
        "n_signal": n_signal,
        "n_noise": n_noise,
        "hit_rate": round(hits / n_signal, 4) if n_signal else np.nan,
        "fa_rate": round(fas / n_noise, 4) if n_noise else np.nan,
        "d_hat": round(d_hat, 4) if not np.isnan(d_hat) else np.nan,
        "type2_auc": round(type2_auc, 4),
        "d_star": round(d_star, 4),
        "mc": round(mc, 4) if not np.isnan(mc) else np.nan,
    }


def bootstrap_stats(df: pd.DataFrame, n_boot: int = 2000, rng_seed: int = 42) -> dict:
    """Bootstrap CIs on hit_rate, fa_rate, d_hat, type-2 AUC, and MC."""
    rng = np.random.default_rng(rng_seed)
    boots = {"d_hat": [], "type2_auc": [], "mc": []}
    n = len(df)
    if n < 10:
        return {k: (np.nan, np.nan) for k in boots}

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        samp = df.iloc[idx]
        stats = compute_metacog_stats(samp)
        for k in boots:
            v = stats[k]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                boots[k].append(v)

    out = {}
    for k, vals in boots.items():
        if len(vals) < 100:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = (round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4))
    return out


def build_scatter_figure(stats: pd.DataFrame, out_path: Path) -> None:
    """Scatter of d_hat vs type-2 AUC, one point per model."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # Reasoning vs non-reasoning coloring
    stats_reas = stats[stats["is_reasoning"].astype(bool)]
    stats_non = stats[~stats["is_reasoning"].astype(bool)]

    for df_sub, color, label in [
        (stats_non, "#4c72b0", "non-reasoning"),
        (stats_reas, "#c44e52", "reasoning-enhanced"),
    ]:
        if df_sub.empty:
            continue
        ax.errorbar(
            df_sub["d_hat"], df_sub["type2_auc"],
            xerr=[df_sub["d_hat"] - df_sub["d_hat_ci_low"],
                  df_sub["d_hat_ci_high"] - df_sub["d_hat"]],
            yerr=[df_sub["type2_auc"] - df_sub["type2_auc_ci_low"],
                  df_sub["type2_auc_ci_high"] - df_sub["type2_auc"]],
            fmt="o", color=color, label=label, capsize=3, alpha=0.85,
        )
        for _, row in df_sub.iterrows():
            ax.annotate(row["model"], (row["d_hat"], row["type2_auc"]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5,
               label="chance-level metacognition")
    ax.set_xlabel(r"$\hat{d}$ (object-level discrimination)")
    ax.set_ylabel("Type-2 AUC (metacognitive discrimination)")
    ax.set_title("Object-level vs metacognitive discrimination on Task 11\n"
                 "(reasoning models at high $\\hat{d}$ with ~0.5 type-2 AUC "
                 "indicate metacognitive inefficiency)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.3, 1.0)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--task-id", default="t11")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    catalog = pd.read_csv(args.catalog)
    task_rows = catalog[catalog["task_id"] == args.task_id]

    results = []
    for _, cat_row in task_rows.iterrows():
        csv_path = cat_row.get("csv_file_path")
        if not isinstance(csv_path, str) or not Path(csv_path).exists():
            print(f"Skipping {cat_row['model']}: no CSV found at {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Could not read CSV for {cat_row['model']}: {e}")
            continue

        if any(c not in df.columns for c in COLUMN_MAP.values()):
            missing = [c for c in COLUMN_MAP.values() if c not in df.columns]
            print(f"Skipping {cat_row['model']}: missing columns {missing}")
            continue

        # Restore raw confidence values from raw_response if available.
        # Kaggle preprocessing clamps the `confidence` column to 1-4, which
        # destroys metacognitive variance for models that used a wider scale
        # (e.g. claude-haiku-4-5 used 4-9 but is binned to constant 4).
        # Falls back to the binned column for rows that cannot be parsed.
        if "raw_response" in df.columns:
            import re as _re
            _conf_col = COLUMN_MAP["confidence"]
            def _extract_raw_conf(s):
                if not isinstance(s, str):
                    return None
                m = _re.search(r'"confidence"\s*:\s*(\d+(?:\.\d+)?)', s)
                return float(m.group(1)) if m else None
            raw_confs = df["raw_response"].apply(_extract_raw_conf)
            n_parsed = int(raw_confs.notna().sum())
            n_total = len(df)
            if n_parsed > 0:
                df[_conf_col] = raw_confs.where(raw_confs.notna(), df[_conf_col])
                _raw_unique = int(raw_confs.dropna().nunique())
                _raw_min = raw_confs.dropna().min()
                _raw_max = raw_confs.dropna().max()
                print(f"  {cat_row['model']}: parsed raw conf for {n_parsed}/{n_total} trials "
                      f"({_raw_unique} unique values, range {_raw_min}-{_raw_max})")
            else:
                print(f"  {cat_row['model']}: raw_response unparseable; using binned confidence")

        stats = compute_metacog_stats(df)
        boots = bootstrap_stats(df, n_boot=args.n_boot)

        results.append({
            "model": cat_row["model"],
            "is_reasoning": cat_row["is_reasoning"],
            **stats,
            "d_hat_ci_low": boots["d_hat"][0],
            "d_hat_ci_high": boots["d_hat"][1],
            "type2_auc_ci_low": boots["type2_auc"][0],
            "type2_auc_ci_high": boots["type2_auc"][1],
            "mc_ci_low": boots["mc"][0],
            "mc_ci_high": boots["mc"][1],
        })

    if not results:
        print(f"No {args.task_id} data could be analyzed. "
              f"Check CSV paths and required columns.")
        return

    out_df = pd.DataFrame(results).sort_values("d_hat", ascending=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out_csv, index=False)

    print(f"Analyzed {len(out_df)} models on {args.task_id}.")
    print()
    print(out_df[[
        "model", "is_reasoning", "n_trials", "hit_rate", "fa_rate",
        "d_hat", "type2_auc", "mc",
    ]].to_string(index=False))

    build_scatter_figure(out_df, args.out_fig)
    print(f"\nStats CSV: {args.out_csv}")
    print(f"Figure:    {args.out_fig}")


if __name__ == "__main__":
    main()
