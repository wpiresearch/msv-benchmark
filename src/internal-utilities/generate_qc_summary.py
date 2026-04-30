#!/usr/bin/env python3
"""
generate_qc_summary.py
======================

Generates a per-model QC summary table from forced-answer Phase 1 output
CSVs, applying the frozen QC rule documented in QC_RULE_v1.md.

Background
----------
Forced-answer Phase 1 outputs may contain protocol-compliance failures
(parse failures), confidence-scale-use degeneracies (constant confidence),
and answer-option anchoring (one letter dominates). These are evaluation-
relevant findings, not measurement noise. The QC summary records them
in an auditable CSV that should be saved BEFORE any inferential bootstrap
runs, so that the inclusion criterion for the primary forced-answer
ranking is committed before the inferential analysis can be tuned.

The QC rule (frozen as QC_RULE_v1.md):
- Hard QC pass: valid >= 70/80 AND parse_fail <= 10/80
- Soft flags (non-exclusionary):
  * answer_anchor: max share of any single answer letter > 0.55
  * confidence_collapse: max share of any single confidence level > 0.90
  * low_conf_scale_use: only 2 of 4 confidence levels used at all

Usage
-----
    python generate_qc_summary.py \\
        --forced_answer_dir results/results-gpqa-2026-03-25/forced_answer_phase1/ \\
        --output_csv        results/results-gpqa-2026-03-25/qc_summary.csv

Inputs
------
- forced_answer_dir: directory of per-model forced-answer Phase 1 CSVs.
  Each CSV must have columns: question_id, answer, confidence,
  parse_failure (and optionally correct, raw_response).
  Non-model CSVs (e.g., qc_summary.csv itself) are skipped by checking
  for the required columns.

Output
------
- output_csv: per-model QC table with columns:
    model, n_total, valid, parse_fail, max_answer_share,
    max_conf_share, n_conf_levels, hard_qc, soft_flags
  Rows are sorted by model name.

Also prints the table to stdout for quick inspection.

The output CSV is intended to be saved to a stable location BEFORE the
forced-answer bootstrap is submitted. Its modification time becomes the
audit trail for "QC rule was committed before inferential analysis."

Dependencies
------------
stdlib only (csv, collections).
"""

import argparse
import collections
import csv
import glob
import os
import sys


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--forced_answer_dir", required=True,
                    help="Directory of forced-answer Phase 1 per-model CSVs")
    ap.add_argument("--output_csv", required=True,
                    help="Where to write QC summary CSV")
    ap.add_argument("--hard_qc_min_valid", type=int, default=70,
                    help="Hard QC: minimum valid responses out of 80 "
                         "(default: 70, frozen rule)")
    ap.add_argument("--hard_qc_max_parse_fail", type=int, default=10,
                    help="Hard QC: maximum parse failures (default: 10)")
    ap.add_argument("--answer_anchor_threshold", type=float, default=0.55,
                    help="Soft flag if any answer letter has share above "
                         "this threshold among valid responses (default: 0.55)")
    ap.add_argument("--conf_collapse_threshold", type=float, default=0.90,
                    help="Soft flag if any confidence level has share above "
                         "this threshold among valid responses (default: 0.90)")
    ap.add_argument("--low_conf_levels_threshold", type=int, default=2,
                    help="Soft flag if number of confidence levels actually "
                         "used is at or below this threshold (default: 2)")
    args = ap.parse_args()

    if not os.path.exists(args.forced_answer_dir):
        sys.exit(f"ERROR: forced_answer_dir does not exist: {args.forced_answer_dir}")

    rows = []
    csv_files = sorted(glob.glob(os.path.join(args.forced_answer_dir, "*.csv")))
    if not csv_files:
        sys.exit(f"ERROR: no CSVs in {args.forced_answer_dir}")

    for fpath in csv_files:
        model = os.path.splitext(os.path.basename(fpath))[0]
        n = parse_fail = 0
        ans = collections.Counter()
        conf = collections.Counter()
        with open(fpath) as f:
            r = csv.DictReader(f)
            # Skip non-model CSVs (qc_summary, etc.)
            if not r.fieldnames or "question_id" not in r.fieldnames:
                continue
            if "confidence" not in r.fieldnames:
                continue
            for row in r:
                n += 1
                pf = row.get("parse_failure", "") in ("True", "true", "1")
                if pf:
                    parse_fail += 1
                    continue
                ans[row.get("answer", "")] += 1
                conf[row.get("confidence", "")] += 1
        valid = n - parse_fail
        max_answer_share = (max(ans.values()) / valid) if valid else 1.0
        max_conf_share = (max(conf.values()) / valid) if valid else 1.0
        n_conf_levels = len([k for k, v in conf.items() if k and v > 0])
        hard_qc = ("pass" if (valid >= args.hard_qc_min_valid
                              and parse_fail <= args.hard_qc_max_parse_fail)
                   else "fail")
        flags = []
        if max_answer_share > args.answer_anchor_threshold:
            flags.append("answer_anchor")
        if max_conf_share > args.conf_collapse_threshold:
            flags.append("confidence_collapse")
        if n_conf_levels <= args.low_conf_levels_threshold:
            flags.append("low_conf_scale_use")
        rows.append({
            "model": model,
            "n_total": n,
            "valid": valid,
            "parse_fail": parse_fail,
            "max_answer_share": f"{max_answer_share:.3f}",
            "max_conf_share": f"{max_conf_share:.3f}",
            "n_conf_levels": n_conf_levels,
            "hard_qc": hard_qc,
            "soft_flags": "|".join(flags) if flags else "ok",
        })

    if not rows:
        sys.exit(f"ERROR: no model CSVs found in {args.forced_answer_dir} "
                 f"(non-model files like qc_summary.csv are skipped)")

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote QC summary to {args.output_csv}\n")
    print(f"{'model':25s} {'valid':>5} {'parse':>5} "
          f"{'ans%':>6} {'conf%':>6} {'lvls':>4} {'hard_qc':>8}  flags")
    print("-" * 80)
    for r in rows:
        print(f"{r['model']:25s} {r['valid']:>5d} {r['parse_fail']:>5d} "
              f"{float(r['max_answer_share']):>6.3f} "
              f"{float(r['max_conf_share']):>6.3f} "
              f"{r['n_conf_levels']:>4d} {r['hard_qc']:>8s}  {r['soft_flags']}")

    n_pass = sum(1 for r in rows if r["hard_qc"] == "pass")
    n_fail = len(rows) - n_pass
    print(f"\nSummary: {n_pass} pass, {n_fail} fail (out of {len(rows)} models)")


if __name__ == "__main__":
    main()
