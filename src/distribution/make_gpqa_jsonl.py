#!/usr/bin/env python3
"""
Convert a GPQA Diamond CSV (as produced by prepare_gpqa.py) into the
JSONL format that run_forced_answer_phase1_turing.py expects.

Output JSONL schema (one JSON object per line):
    question_id, question, option_a, option_b, option_c, option_d,
    correct_answer  (letter A/B/C/D)

Usage:
    python scripts/make_gpqa_jsonl.py \\
        --input  ~/msv_benchmark/data/gpqa_sampled_200.csv \\
        --output data/gpqa_diamond_80.jsonl \\
        --question_ids data/kaggle_extracted/per_task/t01_delegate_game.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REQUIRED = ["question_id", "question", "option_a", "option_b",
            "option_c", "option_d", "correct_answer"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="Source CSV (from prepare_gpqa.py)")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination JSONL path")
    ap.add_argument("--question_ids", type=Path, default=None,
                    help="Optional: CSV with a 'question_id' column to "
                         "filter by (e.g. the Kaggle t01 file). Ensures the "
                         "Turing run uses the SAME 80 questions as the "
                         "Kaggle run.")
    ap.add_argument("--n", type=int, default=None,
                    help="Limit to first N rows after filtering "
                         "(default: all)")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"ERROR: {args.input} does not exist")

    # Optionally load the set of question_ids to filter to
    keep_ids = None
    if args.question_ids is not None:
        if not args.question_ids.exists():
            sys.exit(f"ERROR: {args.question_ids} does not exist")
        with args.question_ids.open() as f:
            r = csv.DictReader(f)
            if "question_id" not in (r.fieldnames or []):
                sys.exit(f"ERROR: {args.question_ids} has no 'question_id' column")
            keep_ids = {row["question_id"] for row in r}
        print(f"Filtering to {len(keep_ids)} unique question_ids from "
              f"{args.question_ids}")

    rows_out = []
    skipped = 0
    with args.input.open() as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = set(REQUIRED) - cols
        if missing:
            sys.exit(f"ERROR: input CSV missing required columns: {missing}\n"
                     f"Available columns: {sorted(cols)}")
        for row in reader:
            if keep_ids is not None and row["question_id"] not in keep_ids:
                skipped += 1
                continue
            ca = str(row["correct_answer"]).strip().upper()
            if ca not in ("A", "B", "C", "D"):
                sys.exit(f"ERROR: question_id={row['question_id']!r} has "
                         f"correct_answer={ca!r}, expected A/B/C/D")
            rows_out.append({
                "question_id":    row["question_id"],
                "question":       row["question"],
                "option_a":       row["option_a"],
                "option_b":       row["option_b"],
                "option_c":       row["option_c"],
                "option_d":       row["option_d"],
                "correct_answer": ca,
            })
            if args.n is not None and len(rows_out) >= args.n:
                break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(rows_out)} rows to {args.output} "
          f"(skipped {skipped} rows that were not in the question_ids filter)")


if __name__ == "__main__":
    main()
