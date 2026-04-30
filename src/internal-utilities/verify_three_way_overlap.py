#!/usr/bin/env python3
"""
verify_three_way_overlap.py
===========================

Pre-flight check before submitting step 4b: confirms that the three-way
intersection of question_ids across the Phase 2 institutional cohort,
the forced-answer Phase 1 output, and the Kaggle subset specification
is exactly the expected size for every model.

Background
----------
Step 4b's bootstrap consumes Phase 2 behavioral data (filtered to 80
questions) and forced-answer Phase 1 declarative data (80 questions).
For the cross-protocol comparison to honor the same-item claim, the
three-way intersection (Phase 2 panel ∩ forced-answer panel ∩ Kaggle
filter) must equal the expected n for every model.

If any model's three-way overlap is smaller than expected, the bootstrap
will silently proceed with reduced data for that model, producing
mixed-cohort comparisons. This script exits non-zero so it can be wired
into a Makefile or CI check.

Usage
-----
    python verify_three_way_overlap.py \\
        --phase2_dir         results/reproduced/turing_analysis_input/delegate_game/ \\
        --forced_answer_dir  results/results-gpqa-2026-03-25/forced_answer_phase1/ \\
        --kaggle_subset_csv  results/gpqa_kaggle_candidates.csv \\
        --expected_n         80

Inputs
------
- phase2_dir: directory of per-model Phase 2 CSVs (canonical schema with
  question_id column). Output of adapt_turing_data.py.
- forced_answer_dir: directory of per-model forced-answer CSVs.
- kaggle_subset_csv: CSV with question_id column defining the subset.
- expected_n: the size each three-way intersection must equal.

Output
------
- Per-model table to stdout with phase2 panel size, forced-answer panel
  size, three-way intersection, and OK/PARTIAL status.
- Exit 0 if every model has three_way_overlap == expected_n.
- Exit 2 if any model has a partial overlap.
- Exit 3 if any model is missing from one side or the other.

Dependencies
------------
stdlib only (csv).
"""

import argparse
import csv
import glob
import os
import sys


def load_qids_from_csv(path):
    qids = set()
    with open(path) as f:
        r = csv.DictReader(f)
        if "question_id" not in (r.fieldnames or []):
            return qids
        for row in r:
            qids.add(row["question_id"])
    return qids


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--phase2_dir", required=True,
                    help="Per-model Phase 2 CSVs (after adapt_turing_data.py)")
    ap.add_argument("--forced_answer_dir", required=True,
                    help="Per-model forced-answer Phase 1 CSVs")
    ap.add_argument("--kaggle_subset_csv", required=True,
                    help="CSV with question_id column defining the subset")
    ap.add_argument("--expected_n", type=int, default=80,
                    help="Expected size of three-way intersection per model "
                         "(default: 80, the Kaggle subset size)")
    args = ap.parse_args()

    if not os.path.exists(args.kaggle_subset_csv):
        sys.exit(f"ERROR: kaggle_subset_csv does not exist: {args.kaggle_subset_csv}")
    kaggle_qids = load_qids_from_csv(args.kaggle_subset_csv)
    print(f"Kaggle subset: {len(kaggle_qids)} question_ids "
          f"from {args.kaggle_subset_csv}")

    if not os.path.exists(args.phase2_dir):
        sys.exit(f"ERROR: phase2_dir does not exist: {args.phase2_dir}")
    if not os.path.exists(args.forced_answer_dir):
        sys.exit(f"ERROR: forced_answer_dir does not exist: {args.forced_answer_dir}")

    print(f"\n{'model':30s}  {'phase2':>6}  {'forced_answer':>13}  "
          f"{'three_way':>9}  status")
    print("-" * 80)

    fa_files = sorted(glob.glob(os.path.join(args.forced_answer_dir, "*.csv")))
    if not fa_files:
        sys.exit(f"ERROR: no CSVs in {args.forced_answer_dir}")

    any_problem = False
    any_missing = False

    for fa_path in fa_files:
        model = os.path.splitext(os.path.basename(fa_path))[0]

        # Skip non-model CSVs (qc_summary.csv etc.)
        with open(fa_path) as f:
            r = csv.DictReader(f)
            if "question_id" not in (r.fieldnames or []):
                continue
            fa_qids = set(row["question_id"] for row in r)

        p2_path = os.path.join(args.phase2_dir, model + ".csv")
        if not os.path.exists(p2_path):
            print(f"{model:30s}  {'?':>6}  {len(fa_qids):>13}  {'?':>9}  "
                  f"MISSING_PHASE2")
            any_missing = True
            continue

        p2_qids = load_qids_from_csv(p2_path)
        three_way = kaggle_qids & p2_qids & fa_qids

        status = "OK" if len(three_way) == args.expected_n \
            else f"PARTIAL ({len(three_way)} != {args.expected_n})"
        if len(three_way) != args.expected_n:
            any_problem = True

        print(f"{model:30s}  {len(p2_qids):>6}  {len(fa_qids):>13}  "
              f"{len(three_way):>9}  {status}")

    print()
    if any_missing:
        print("FAIL: at least one model was missing from one side", file=sys.stderr)
        sys.exit(3)
    if any_problem:
        print(f"FAIL: at least one model has three-way overlap != "
              f"{args.expected_n}. Investigate before filtering and "
              f"submitting bootstrap.", file=sys.stderr)
        sys.exit(2)
    print(f"OK: all models have three-way overlap = {args.expected_n}. "
          f"Safe to filter and proceed.")


if __name__ == "__main__":
    main()
