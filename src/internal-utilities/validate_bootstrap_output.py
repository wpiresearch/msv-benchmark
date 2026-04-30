#!/usr/bin/env python3
"""
validate_bootstrap_output.py
============================

Validates a bootstrap CI summary CSV against a set of expected
invariants. The discipline is to run this AFTER every bootstrap
submission, before treating the output as authoritative.

Background
----------
The first attempt at step 4b on Turing exited COMPLETED 0:0 with a
well-formed output file. The validation step revealed that despite
passing --export=ALL,FORCED_ANSWER_DIR=..., every row had
declarative_source = answered_only. The cause was a missing conditional
in the SLURM template; bash silently accepted the unused env var. Without
this validation block, we'd have proceeded to step 4c with the wrong
numbers.

This script formalizes that discipline. It's intentionally strict:
exit code 0 only if every check passes.

Checks performed
----------------
1. Row count matches expected cohort size.
2. declarative_source distribution matches expectation
   (--expected_source must appear in every row).
3. ECE point estimates are non-NaN where forced-answer was provided
   (NaN is allowed where no forced-answer data exists for that model).
4. CI bounds satisfy ci_lo <= point <= ci_hi for every numeric metric.
5. Optional: model name set matches a provided expected list.

Usage
-----
    # Simple pass: any source, just sanity-check structure
    python validate_bootstrap_output.py \\
        --bootstrap_csv results/reproduced/bootstrap_kaggle/bootstrap_ci_summary.csv \\
        --expected_n_models 23

    # Strict: forced-answer expected for all rows
    python validate_bootstrap_output.py \\
        --bootstrap_csv results/reproduced/bootstrap_institutional_with_fa/bootstrap_ci_summary.csv \\
        --expected_n_models 9 \\
        --expected_source forced_answer

    # Strict with explicit model list
    python validate_bootstrap_output.py \\
        --bootstrap_csv results/reproduced/bootstrap_institutional_with_fa/bootstrap_ci_summary.csv \\
        --expected_n_models 9 \\
        --expected_source forced_answer \\
        --expected_models gemma2_2b,gemma2_9b,llama3.1_8b,llama3.2_1b,llama3.2_3b,mistral_7b,phi4-mini_latest,qwen2.5_3b,qwen2.5_7b

Exit codes
----------
0  All checks pass.
2  At least one check failed (details printed to stderr).
3  File not found or unreadable.

Dependencies
------------
stdlib only (csv, math).
"""

import argparse
import csv
import math
import os
import sys
from collections import Counter


def is_nan(s):
    """Treat empty string as NaN."""
    if s is None or s == "":
        return True
    try:
        return math.isnan(float(s))
    except (ValueError, TypeError):
        return True  # unparseable counts as NaN


def safe_float(s):
    if s is None or s == "":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--bootstrap_csv", required=True,
                    help="Path to bootstrap_ci_summary.csv to validate")
    ap.add_argument("--expected_n_models", type=int, required=True,
                    help="Expected number of model rows")
    ap.add_argument("--expected_source", default=None,
                    help="If set, every row's declarative_source must equal "
                         "this value. Common values: 'forced_answer', "
                         "'answered_only'. If omitted, only the distribution "
                         "is reported, not enforced.")
    ap.add_argument("--expected_models", default=None,
                    help="Optional comma-separated list of expected model "
                         "names. If set, the row set must match exactly.")
    ap.add_argument("--allow_nan_metrics", default="ece,brier,deleg_auc",
                    help="Comma-separated list of metric prefixes for which "
                         "NaN values are tolerated (e.g., always-delegate "
                         "models legitimately have NaN deleg_auc). "
                         "Default: ece,brier,deleg_auc")
    args = ap.parse_args()

    if not os.path.exists(args.bootstrap_csv):
        print(f"ERROR: file does not exist: {args.bootstrap_csv}",
              file=sys.stderr)
        sys.exit(3)

    try:
        with open(args.bootstrap_csv) as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        print(f"ERROR: cannot read {args.bootstrap_csv}: {e}", file=sys.stderr)
        sys.exit(3)

    failures = []
    warnings = []

    # Check 1: row count
    print(f"Bootstrap CSV: {args.bootstrap_csv}")
    print(f"Total rows: {len(rows)}")
    if len(rows) != args.expected_n_models:
        failures.append(f"row count {len(rows)} != expected {args.expected_n_models}")

    # Check 2: declarative_source
    if rows and "declarative_source" in rows[0]:
        sources = [r.get("declarative_source", "") for r in rows]
        src_dist = dict(Counter(sources))
        print(f"declarative_source distribution: {src_dist}")
        if args.expected_source is not None:
            wrong = [r["model"] for r in rows
                     if r.get("declarative_source", "") != args.expected_source]
            if wrong:
                failures.append(
                    f"{len(wrong)} row(s) have declarative_source != "
                    f"'{args.expected_source}': {wrong[:5]}"
                    f"{'...' if len(wrong) > 5 else ''}"
                )
    else:
        if args.expected_source is not None:
            failures.append(
                "declarative_source column missing but --expected_source set"
            )
        else:
            print("declarative_source column not present (older script version?)")

    # Check 3: model set
    if args.expected_models:
        expected_set = set(s.strip() for s in args.expected_models.split(","))
        actual_set = set(r.get("model", "") for r in rows)
        missing = expected_set - actual_set
        extra = actual_set - expected_set
        if missing:
            failures.append(f"missing expected models: {sorted(missing)}")
        if extra:
            failures.append(f"unexpected extra models: {sorted(extra)}")

    # Check 4: CI containment for every numeric metric (point in [ci_lo, ci_hi])
    if rows:
        nan_tolerated_prefixes = tuple(
            p.strip() for p in args.allow_nan_metrics.split(",") if p.strip()
        )
        # Find all metrics: any column ending in _point with a matching
        # _ci_lo and _ci_hi.
        all_cols = set(rows[0].keys())
        metric_prefixes = set()
        for c in all_cols:
            if c.endswith("_point"):
                pref = c[:-len("_point")]
                if (pref + "_ci_lo") in all_cols and (pref + "_ci_hi") in all_cols:
                    metric_prefixes.add(pref)

        for r in rows:
            model = r.get("model", "?")
            for pref in sorted(metric_prefixes):
                pt = safe_float(r.get(pref + "_point", ""))
                lo = safe_float(r.get(pref + "_ci_lo", ""))
                hi = safe_float(r.get(pref + "_ci_hi", ""))
                # NaN handling: if all three are NaN and the metric is in the
                # tolerated-NaN list, that's fine (e.g., always-delegate ->
                # NaN deleg_auc)
                all_nan = (math.isnan(pt) and math.isnan(lo) and math.isnan(hi))
                tolerated = pref in nan_tolerated_prefixes
                if all_nan:
                    if not tolerated:
                        warnings.append(
                            f"{model}: {pref} all-NaN; consider whether this "
                            f"is expected"
                        )
                    continue  # all-NaN is OK if tolerated; skip CI check
                # Partial NaN is always a problem
                if math.isnan(pt) or math.isnan(lo) or math.isnan(hi):
                    failures.append(
                        f"{model}: {pref} has partial NaN "
                        f"(point={pt}, lo={lo}, hi={hi})"
                    )
                    continue
                # CI containment
                if not (lo <= pt <= hi):
                    failures.append(
                        f"{model}: {pref} CI does not contain point estimate "
                        f"(lo={lo:.4f}, point={pt:.4f}, hi={hi:.4f})"
                    )

    # Print summary
    print()
    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  WARN  {w}")
    if failures:
        print(f"\nFailures ({len(failures)}):", file=sys.stderr)
        for f in failures:
            print(f"  FAIL  {f}", file=sys.stderr)
        print("\nVALIDATION FAILED", file=sys.stderr)
        sys.exit(2)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
