"""
Script 0: Build a catalog of all available Kaggle run files and CSVs.

This script scans the data directory for Kaggle run files (*.run.json) and
per-question CSVs (tXX_*_results.csv), extracts the model slug and task ID
from each filename, and produces a manifest CSV that all subsequent scripts
use as their input.

Expected filename patterns:
  Run files: tXX-<task_name>-run_id_Run_1_<provider>_<model_slug>.run.json
             (provider and model_slug separated by underscores, e.g.
              anthropic_claude-haiku-4-5-20251001,
              google_gemini-2.5-flash,
              zai_glm-5)
  CSVs:      tXX_<task_shortname>_results.csv
             (saved by each task's code in /kaggle/working/; you will need
              to rename these to include the model slug when copying off
              Kaggle, e.g. t01_delegate_game_results_<model>.csv)

If your filenames diverge from these patterns, adjust the regex patterns
below. The rest of the pipeline depends on the model slug being consistent
across files for the same model.

Inputs:
  --run-dir: directory containing *.run.json files
  --csv-dir: directory containing *_results*.csv files

Outputs:
  --out: path to write run_catalog.csv

Usage:
  python script_0_catalog.py \\
      --run-dir data/kaggle_runs \\
      --csv-dir data/kaggle_csvs \\
      --out outputs/run_catalog.csv
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd

# Known task short-names and their expected total question counts
# Task 11 (MC Binary Pairs) has 160 prompts (2 per question); all others are 80.
TASK_TOTALS = {
    "t01": ("delegate_game", 80),
    "t02": ("declared_probe", 80),
    "t03": ("second_chance", 80),
    "t04": ("confidence_entropy", 80),
    "t05": ("teammate_delegate", 80),
    "t06": ("behavioral_er", 80),
    "t07": ("behavioral_ci", 80),
    "t08": ("behavioral_em", 80),
    "t09": ("behavioral_pi", 80),
    "t10": ("dpp_sequence", 80),  # 400 prompts (5 per question)
    "t11": ("mc_binary_pairs", 160),
}

# Hand-labeled reasoning-model flag. Update this when new models are added.
# Used downstream by Scripts 5 and 6 to group models by category.
REASONING_MODELS = {
    "deepseek-r1-0528",
    "gemini-2.5-flash",
    "glm-5",
    "gemma-4-31b-it",  # Gemma 4 behaves like a reasoning model on these tasks
    "qwen3-next-80b-a3b-thinking",
    "claude-opus-4-1-20250805",  # Claude with extended thinking
}

RUN_FILE_RE = re.compile(
    r"(?P<task_id>t\d{2})-msv_(?P<task_name>[a-z_]+)-run_id_Run_\d+_"
    r"(?P<provider>[a-z\-]+)_(?P<model>[a-zA-Z0-9\-.]+?)\.run\.json$"
)

CSV_FILE_RE = re.compile(
    r"(?P<task_id>t\d{2})_(?P<task_name>[a-z_]+)_results"
    r"(?:_(?P<model>[a-zA-Z0-9\-.]+))?\.csv$"
)


def parse_run_file(path: Path) -> dict | None:
    """Extract model, task, and request count from a run file."""
    m = RUN_FILE_RE.search(path.name)
    if not m:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        reqs = data.get("conversations", [{}])[0].get("requests", [])
        n_requests = len(reqs)
    except (json.JSONDecodeError, KeyError, IndexError):
        n_requests = 0
    task_id = m.group("task_id")
    total_expected = TASK_TOTALS.get(task_id, (None, 80))[1]
    return {
        "model": m.group("model"),
        "provider": m.group("provider"),
        "task_id": task_id,
        "task_name": m.group("task_name"),
        "n_requests": n_requests,
        "n_expected": total_expected,
        "completion_rate": round(n_requests / total_expected, 4) if total_expected else None,
        "run_file_path": str(path),
        "is_reasoning": m.group("model") in REASONING_MODELS,
    }


def parse_csv_file(path: Path) -> dict | None:
    """Extract model and task from a CSV filename."""
    m = CSV_FILE_RE.search(path.name)
    if not m:
        return None
    return {
        "task_id": m.group("task_id"),
        "model": m.group("model") or "UNKNOWN",
        "csv_file_path": str(path),
    }


def build_catalog(run_dir: Path, csv_dir: Path) -> pd.DataFrame:
    """Walk both directories and build a joined catalog."""
    run_rows = []
    for p in sorted(run_dir.glob("*.run.json")):
        row = parse_run_file(p)
        if row:
            run_rows.append(row)

    csv_rows = []
    for p in sorted(csv_dir.glob("*.csv")):
        row = parse_csv_file(p)
        if row:
            csv_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    csv_df = pd.DataFrame(csv_rows)

    if run_df.empty:
        print(f"WARNING: No run files matched in {run_dir}")
        return pd.DataFrame()

    # Left-join CSV paths onto run-file records
    if not csv_df.empty:
        merged = run_df.merge(csv_df, on=["task_id", "model"], how="left")
    else:
        merged = run_df.copy()
        merged["csv_file_path"] = None

    return merged.sort_values(["model", "task_id"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--csv-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    catalog = build_catalog(args.run_dir, args.csv_dir)
    if catalog.empty:
        print("No entries found. Check directory paths and filename patterns.")
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(args.out, index=False)

    print(f"Cataloged {len(catalog)} runs from {catalog['model'].nunique()} models.")
    print(f"Tasks represented: {sorted(catalog['task_id'].unique().tolist())}")
    print()
    print("Summary by model:")
    summary = catalog.groupby("model").agg(
        n_tasks=("task_id", "nunique"),
        total_requests=("n_requests", "sum"),
        reasoning=("is_reasoning", "first"),
    )
    print(summary.to_string())
    print(f"\nCatalog saved to {args.out}")


if __name__ == "__main__":
    main()
