"""
Script 1: Per-model verbosity analysis from Kaggle run files.

Reads the per-request metrics stored in each *.run.json file (inputTokens,
outputTokens, totalBackendLatencyMs) and produces summary statistics that
distinguish reasoning-enhanced models from non-reasoning models on a purely
behavioral signature: how many tokens do they emit per response to a prompt
that explicitly asks for JSON-only output?

This is the first-line diagnostic for the verbose-CoT failure mode. Models
with median output_tokens >> 500 despite JSON-only prompts are reasoning
models that will eventually saturate the Kaggle SDK conversation context
and crash (see README diagnostic note on verbose-CoT context accumulation).

Inputs:
  --catalog: path to run_catalog.csv from Script 0
  --out-csv: path to write verbosity_stats.csv
  --out-fig: path to write verbosity_distribution.png

Outputs:
  verbosity_stats.csv with one row per (model, task) pair:
    - n_requests
    - output_tokens: mean, median, p90, max
    - input_tokens_cumulative_at_end (estimated context saturation)
    - latency_ms_mean
    - verbosity_index (mean output tokens; headline statistic)

  verbosity_distribution.png: horizontal bar chart of mean output tokens
    per model, averaged across all tasks, with models sorted from terse
    to verbose. A vertical line at 500 tokens separates "followed JSON
    instructions" from "emitted verbose CoT."

Usage:
  python script_1_verbosity.py \\
      --catalog outputs/run_catalog.csv \\
      --out-csv outputs/verbosity_stats.csv \\
      --out-fig outputs/verbosity_distribution.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def extract_request_metrics(run_file: Path) -> list[dict]:
    """Return per-request metrics for a single run file."""
    try:
        with open(run_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []

    reqs = data.get("conversations", [{}])[0].get("requests", [])
    rows = []
    cumulative_in = 0
    cumulative_out = 0
    for i, r in enumerate(reqs):
        m = r.get("metrics", {})
        in_tok = int(m.get("inputTokens", 0))
        # outputTokens may be missing from metrics for some providers
        # (e.g., Google Gemma 3 endpoints did not return it via Kaggle).
        # Fall back to character-count/4 approximation from the assistant
        # turn in r["contents"]. This is a conservative estimate; for
        # relative-verbosity comparisons across models it is sufficient.
        if "outputTokens" in m:
            out_tok = int(m["outputTokens"])
        else:
            out_tok = 0
            for content in r.get("contents", []):
                role = str(content.get("role", "")).upper()
                if "ASSISTANT" in role:
                    parts = content.get("parts", [])
                    char_count = sum(
                        len(str(p.get("text", "")))
                        for p in parts
                        if isinstance(p, dict)
                    )
                    out_tok = char_count // 4  # rough char->token approx
                    break
        cumulative_in += in_tok
        cumulative_out += out_tok
        rows.append({
            "request_idx": i,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "latency_ms": int(m.get("totalBackendLatencyMs", 0)),
            "cumulative_input_tokens": cumulative_in,
            "cumulative_output_tokens": cumulative_out,
        })
    return rows


def summarize_run(run_file: Path, model: str, task_id: str) -> dict | None:
    """Produce a single-row summary of one run's verbosity profile."""
    metrics = extract_request_metrics(run_file)
    if not metrics:
        return None
    df = pd.DataFrame(metrics)
    return {
        "model": model,
        "task_id": task_id,
        "n_requests": len(df),
        "output_tokens_mean": round(df["output_tokens"].mean(), 1),
        "output_tokens_median": round(df["output_tokens"].median(), 1),
        "output_tokens_p90": round(df["output_tokens"].quantile(0.90), 1),
        "output_tokens_max": int(df["output_tokens"].max()),
        "cumulative_output_at_end": int(df["cumulative_output_tokens"].iloc[-1]),
        "latency_ms_mean": round(df["latency_ms"].mean(), 1),
        "verbosity_index": round(df["output_tokens"].mean(), 0),
    }


def classify_verbosity(mean_tokens: float) -> str:
    """Three-bucket classifier used in the figure legend."""
    if mean_tokens < 150:
        return "terse (follows JSON-only)"
    elif mean_tokens < 800:
        return "moderate"
    else:
        return "verbose CoT (reasoning)"


def build_verbosity_figure(df_model: pd.DataFrame, out_path: Path) -> None:
    """Horizontal bar chart of mean output tokens per model."""
    df_sorted = df_model.sort_values("mean_output_tokens")
    colors = [
        "#4c72b0" if v < 150 else ("#dd8452" if v < 800 else "#c44e52")
        for v in df_sorted["mean_output_tokens"]
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.35)))
    ax.barh(df_sorted["model"], df_sorted["mean_output_tokens"], color=colors)
    ax.axvline(500, linestyle="--", color="gray", alpha=0.6,
               label="500-token reference (JSON-only expected)")
    ax.set_xlabel("Mean output tokens per response (averaged across tasks)")
    ax.set_ylabel("Model")
    ax.set_title("Model verbosity on JSON-only prompts\n"
                 "(higher values indicate chain-of-thought emission "
                 "despite explicit brevity instructions)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Annotate each bar with its numeric value
    for i, (model, val) in enumerate(zip(df_sorted["model"], df_sorted["mean_output_tokens"])):
        ax.text(val + 50, i, f"{val:.0f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-fig", type=Path, required=True)
    args = ap.parse_args()

    catalog = pd.read_csv(args.catalog)
    rows = []
    for _, cat_row in catalog.iterrows():
        summary = summarize_run(Path(cat_row["run_file_path"]),
                                cat_row["model"], cat_row["task_id"])
        if summary:
            rows.append(summary)

    if not rows:
        print("No run files could be summarized. Check catalog paths.")
        return

    stats_df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(args.out_csv, index=False)

    # Aggregate to one row per model for the figure
    model_agg = stats_df.groupby("model").agg(
        mean_output_tokens=("output_tokens_mean", "mean"),
        n_tasks=("task_id", "nunique"),
    ).reset_index()
    model_agg["category"] = model_agg["mean_output_tokens"].apply(classify_verbosity)

    print(f"Processed {len(stats_df)} runs across {model_agg['model'].nunique()} models.")
    print()
    print("Per-model verbosity (sorted by mean output tokens):")
    print(model_agg.sort_values("mean_output_tokens").to_string(index=False))

    build_verbosity_figure(model_agg, args.out_fig)
    print(f"\nStats CSV: {args.out_csv}")
    print(f"Figure:    {args.out_fig}")


if __name__ == "__main__":
    main()
