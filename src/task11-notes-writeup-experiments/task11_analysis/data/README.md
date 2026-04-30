# Task 11 Workspace Data

This directory contains a working copy of the Kaggle data pre-extracted
into:

- `kaggle_runs/`  -- 253 *.run.json files (one per model × task)
- `kaggle_csvs/`  -- 253 per-task results CSVs

This duplicates `data/kaggle-data/` (the package's canonical data
location). It is kept here for two reasons:

1. The Task 11 audit pipeline (run_all.py + script_0..6) reads from
   this layout (kaggle_runs/ and kaggle_csvs/ as siblings).
2. The pipeline orchestrates dozens of scripts that expect to find
   their input files at known paths relative to the workspace.

To avoid duplication, the Makefile's `reproduce-task11` target extracts
the same data into `results/reproduced/task11_audit/raw_temp/` at
run-time. If you want to re-run the pipeline manually, you can either:

- Use this pre-extracted copy (run from this directory)
- Or extract from `data/kaggle-data/kaggle_raw/outputs_logs_corrected.zip`
  yourself

For shipping the package, the working copy is preserved to support
manual re-runs without requiring the unzip step.
