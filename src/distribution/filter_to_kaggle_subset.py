#!/usr/bin/env python3
"""
filter_to_kaggle_subset.py
==========================

Filters per-model Phase 2 Delegate Game CSVs to the 80-question Kaggle
subset, producing a parallel directory tree for the cross-protocol
forced-answer comparison.

Background
----------
The institutional Phase 2 Delegate Game data covers all 198 GPQA Diamond
questions, but the forced-answer Phase 1 protocol covers only the 80
questions in the Kaggle subset (gpqa_kaggle_candidates.csv). For the
cross-protocol comparison (forced-answer ECE vs. Delegation AUC) to
honor the same-item claim, the bootstrap input must be filtered to the
80-question shared panel before step 4b.

This script does that filter, and writes filter_metadata.json alongside
the output for reproducibility (filter source, expected n, applied-at
timestamp, per-model row counts, null-difficulty counts).

Usage
-----
    python filter_to_kaggle_subset.py \\
        --input_dir   results/reproduced/turing_analysis_input/delegate_game/ \\
        --filter_csv  results/gpqa_kaggle_candidates.csv \\
        --output_dir  results/reproduced/turing_analysis_input_80q/delegate_game/

Inputs
------
- input_dir: directory of per-model CSVs in canonical schema
  (output of adapt_turing_data.py); each CSV has columns
  question_id, answer, correct, confidence, delegated, difficulty.
- filter_csv: CSV with a question_id column whose values define the
  subset to keep. Other columns are ignored.

Outputs
-------
- output_dir: same per-model CSVs, filtered to rows whose question_id
  appears in filter_csv. Filenames preserved.
- output_dir/../filter_metadata.json: metadata recording the filter
  source, expected n, per-model input/output row counts, and per-model
  null-difficulty counts.

The output_dir's parent directory is used for filter_metadata.json so
that the metadata sits one level up from the per-model CSVs (mirroring
the convention that qc_summary.csv sits one level up from forced_answer
CSVs to avoid being globbed as a fake model).

Dependencies
------------
stdlib only (csv, json).
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--input_dir", required=True,
                    help="Directory of per-model Phase 2 CSVs to filter")
    ap.add_argument("--filter_csv", required=True,
                    help="CSV containing question_id values to keep")
    ap.add_argument("--output_dir", required=True,
                    help="Where to write filtered CSVs (will be created)")
    ap.add_argument("--expected_n", type=int, default=80,
                    help="Expected number of question_ids after filter "
                         "(default: 80, the Kaggle subset size). Script "
                         "does not enforce; only records in metadata.")
    args = ap.parse_args()

    # Load filter question_ids
    if not os.path.exists(args.filter_csv):
        sys.exit(f"ERROR: filter CSV does not exist: {args.filter_csv}")
    filter_qids = set()
    with open(args.filter_csv) as f:
        r = csv.DictReader(f)
        if "question_id" not in (r.fieldnames or []):
            sys.exit(f"ERROR: {args.filter_csv} has no 'question_id' column "
                     f"(found: {r.fieldnames})")
        for row in r:
            filter_qids.add(row["question_id"])
    print(f"Loaded {len(filter_qids)} question_ids from {args.filter_csv}")

    if not os.path.exists(args.input_dir):
        sys.exit(f"ERROR: input_dir does not exist: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    metadata = {
        "filter_source": args.filter_csv,
        "expected_n": args.expected_n,
        "filter_applied_at": datetime.now().isoformat(timespec='seconds'),
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "models": {},
    }

    print(f"\n{'model':30s}  {'in':>5}  {'out':>5}  {'difficulty_null':>15}")
    print("-" * 65)
    csv_files = sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    if not csv_files:
        sys.exit(f"ERROR: no CSVs in {args.input_dir}")

    any_size_mismatch = False
    for fpath in csv_files:
        model = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath) as fin:
            reader = csv.DictReader(fin)
            fieldnames = reader.fieldnames
            all_rows = list(reader)
            kept = [r for r in all_rows if r.get("question_id") in filter_qids]
        null_diff = sum(1 for r in kept
                        if not r.get("difficulty") or r["difficulty"] == "")

        out_path = os.path.join(args.output_dir, os.path.basename(fpath))
        with open(out_path, "w") as fout:
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)

        size_flag = ""
        if len(kept) != args.expected_n:
            any_size_mismatch = True
            size_flag = f"  (!= expected {args.expected_n})"

        metadata["models"][model] = {
            "input_rows": len(all_rows),
            "output_rows": len(kept),
            "difficulty_null_count": null_diff,
            "matches_expected": len(kept) == args.expected_n,
        }
        print(f"{model:30s}  {len(all_rows):>5}  {len(kept):>5}  "
              f"{null_diff:>15}{size_flag}")

    # Write metadata one level up from the output_dir (sibling of delegate_game/)
    metadata_dir = os.path.dirname(args.output_dir.rstrip("/")) or "."
    metadata_path = os.path.join(metadata_dir, "filter_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nWrote metadata to {metadata_path}")

    if any_size_mismatch:
        print(f"\nWARNING: at least one model has output rows != expected_n "
              f"({args.expected_n}). Investigate before proceeding to "
              f"bootstrap; partial-panel data will produce mixed-cohort "
              f"comparisons.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
