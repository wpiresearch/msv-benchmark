#!/usr/bin/env python3
"""
diagnose_institutional_inclusion.py
====================================

Pre-flight diagnostic for institutional rank-divergence analysis (step 4c).

Reports which models will actually contribute to a rank-divergence τ
estimate, and whether the resulting cohort size is large enough for
the estimate to be statistically meaningful.

Two scenarios are reported:
  (a) all-attempted: all models with both ECE and Delegation AUC computable
  (b) qc-passing: subset that also passes the frozen forced-answer QC rule
                  (valid >= 70/80, parse_failures <= 10/80)

The qc-passing subset is the appropriate primary inclusion rule for
forced-answer rank-divergence; the all-attempted set serves as a
sensitivity reference.

Background: this diagnostic was developed because step 4c on the
institutional cohort produces a τ estimate from very few models
(after excluding always-delegators with undefined Delegation AUC and
QC-failing models with degraded forced-answer data). For the canonical
9-model cohort the qc-passing mixed-delegator subset is only n=4,
which makes aggregate τ statistically uninformative. The paper therefore
reports per-model results rather than an aggregate institutional τ;
this script makes that decision auditable.

Usage:
    python diagnose_institutional_inclusion.py \\
        --input-dir         results/reproduced/turing_analysis_input_80q/delegate_game/ \\
        --forced-answer-dir results/results-gpqa-2026-03-25/forced_answer_phase1/ \\
        --min-answered      5 \\
        --output-csv        results/reproduced/institutional_inclusion_diagnostic.csv

The --output-csv parameter is optional; if omitted, only the human-readable
summary is printed.

Imports metrics_for_model and load_forced_answer_dir from
compute_rank_divergence_ci.py to use the exact computation the bootstrap
will use, ensuring the diagnostic reflects what step 4c will actually produce.
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

# QC-passing models per the frozen QC rule (see QC_RULE_v1.md):
# Hard QC: valid >= 70/80 AND parse_fail <= 10/80
QC_PASS = {
    "gemma2_9b", "qwen2.5_7b", "mistral_7b", "qwen2.5_3b",
    "llama3.2_3b", "llama3.1_8b", "gemma2_2b",
}
QC_FAIL = {"llama3.2_1b", "phi4-mini_latest"}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of behavioral CSVs (one per model, the 80q-filtered set)",
    )
    ap.add_argument(
        "--forced-answer-dir",
        type=Path,
        required=True,
        help="Directory of forced-answer Phase 1 CSVs",
    )
    ap.add_argument(
        "--min-answered",
        type=int,
        default=5,
        help="min_answered threshold passed to metrics_for_model "
             "(default: 5, matching primary rank-divergence default)",
    )
    ap.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional: write the per-model diagnostic table to this CSV",
    )
    ap.add_argument(
        "--rank-divergence-script",
        type=Path,
        default=None,
        help="Path to compute_rank_divergence_ci.py "
             "(default: ./scripts/compute_rank_divergence_ci.py if present, "
             "else ./compute_rank_divergence_ci.py)",
    )
    args = ap.parse_args()

    # Locate compute_rank_divergence_ci.py and import its helpers
    if args.rank_divergence_script is not None:
        script_path = args.rank_divergence_script
    else:
        for candidate in ("./scripts/compute_rank_divergence_ci.py",
                          "./compute_rank_divergence_ci.py"):
            if Path(candidate).exists():
                script_path = Path(candidate)
                break
        else:
            print(
                "ERROR: could not locate compute_rank_divergence_ci.py.\n"
                "Pass --rank-divergence-script explicitly.",
                file=sys.stderr,
            )
            return 1

    sys.path.insert(0, str(script_path.parent))
    from compute_rank_divergence_ci import (  # noqa: E402
        metrics_for_model, load_forced_answer_dir,
    )

    if not args.input_dir.exists():
        print(f"ERROR: input dir not found: {args.input_dir}", file=sys.stderr)
        return 1
    if not args.forced_answer_dir.exists():
        print(f"ERROR: forced-answer dir not found: {args.forced_answer_dir}",
              file=sys.stderr)
        return 1

    print(f"Loading forced-answer CSVs from {args.forced_answer_dir}")
    fa_dfs = load_forced_answer_dir(args.forced_answer_dir)
    print(f"  loaded {len(fa_dfs)} forced-answer CSVs (post parse-failure drop)")
    for name, fa_df in sorted(fa_dfs.items()):
        print(f"    {name}: {len(fa_df)} valid forced-answer rows")
    print()

    print(f"Loading behavioral CSVs from {args.input_dir}")
    csv_files = sorted(glob.glob(str(args.input_dir / "*.csv")))
    print(f"  found {len(csv_files)} behavioral CSVs")
    print()

    rows = []
    print(
        f"{'model':25s}  {'qc':>5}  {'fa_n':>5}  {'merged':>6}  "
        f"{'deleg_rate':>10}  {'ece':>8}  {'auc':>8}  {'incl':>5}"
    )
    print("-" * 95)

    scenarios = {"all_attempted": [], "qc_passing": []}
    for fp in csv_files:
        name = Path(fp).stem
        df = pd.read_csv(fp)
        fa_df = fa_dfs.get(name)
        m = metrics_for_model(df, args.min_answered, fa_df=fa_df)

        qc = "PASS" if name in QC_PASS else ("FAIL" if name in QC_FAIL else "?")
        fa_n = len(fa_df) if fa_df is not None else 0
        merged_n = m["n_decl"]
        deleg_rate = m["deleg_rate"]
        ece_str = f"{m['ece']:.4f}" if not pd.isna(m["ece"]) else "NaN"
        auc_str = f"{m['deleg_auc']:.4f}" if not pd.isna(m["deleg_auc"]) else "NaN"
        incl = "yes" if (not pd.isna(m["ece"]) and not pd.isna(m["deleg_auc"])) else "no"

        print(
            f"{name:25s}  {qc:>5}  {fa_n:>5d}  {merged_n:>6d}  "
            f"{deleg_rate:>10.4f}  {ece_str:>8}  {auc_str:>8}  {incl:>5}"
        )

        rows.append({
            "model": name,
            "qc_status": qc,
            "fa_valid_rows": fa_n,
            "merged_rows": merged_n,
            "deleg_rate": deleg_rate,
            "ece": m["ece"],
            "deleg_auc": m["deleg_auc"],
            "rank_divergence_eligible": incl == "yes",
        })

        if incl == "yes":
            scenarios["all_attempted"].append(name)
            if name in QC_PASS:
                scenarios["qc_passing"].append(name)

    print()
    print("=== Effective sample sizes ===")
    print(
        f"All-attempted (no QC filter): n = {len(scenarios['all_attempted'])}"
    )
    print(f"  models: {scenarios['all_attempted']}")
    print()
    print(
        f"QC-passing only (frozen QC rule): n = {len(scenarios['qc_passing'])}"
    )
    print(f"  models: {scenarios['qc_passing']}")
    print()

    if len(scenarios["qc_passing"]) < 6:
        print(
            "WARNING: QC-passing subset is below n=6. Aggregate τ on this "
            "cohort will have a wide bootstrap CI and is unlikely to be "
            "statistically informative as a primary result. Consider "
            "reporting only per-model values rather than an aggregate τ."
        )

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nWrote per-model diagnostic table: {args.output_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
