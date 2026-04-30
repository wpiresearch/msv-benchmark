"""
Orchestrator: run all analysis scripts in order.

Executes Scripts 0 through 6 in the correct dependency order. Each script
reads outputs of earlier scripts, so they must be run in sequence (with
Scripts 5 and 6 depending on the outputs of 1-4).

Usage:
  python run_all.py --data-root data --output-root outputs

This assumes the following layout beneath --data-root:
  kaggle_runs/    (*.run.json files)
  kaggle_csvs/    (tXX_*_results*.csv files)

Outputs are written to --output-root. If any step fails, the error is
printed and the orchestrator continues with the remaining steps that
do not depend on the failed step.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], step_name: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  STEP: {step_name}")
    print(f"  CMD:  {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd, check=False)
    ok = result.returncode == 0
    print(f"\n  --> {'OK' if ok else 'FAILED'} (exit code {result.returncode})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--output-root", type=Path, default=Path("outputs"))
    ap.add_argument("--python", default=sys.executable,
                    help="Python interpreter to use (default: current)")
    args = ap.parse_args()

    scripts_dir = Path(__file__).parent
    run_dir = args.data_root / "kaggle_runs"
    csv_dir = args.data_root / "kaggle_csvs"
    args.output_root.mkdir(parents=True, exist_ok=True)

    catalog_csv = args.output_root / "run_catalog.csv"
    verbosity_csv = args.output_root / "verbosity_stats.csv"
    verbosity_fig = args.output_root / "verbosity_distribution.png"
    t11_csv = args.output_root / "task11_metacognitive_efficiency.csv"
    t11_fig = args.output_root / "d_hat_vs_type2auc.png"
    t1_csv = args.output_root / "task1_delegation_curves.csv"
    t1_fig = args.output_root / "delegation_by_difficulty.png"
    t1_slopes = args.output_root / "task1_delegation_curves_slopes.csv"
    t2_csv = args.output_root / "task2_coherence.csv"
    t2_fig = args.output_root / "task2_coherence_scatter.png"
    conv_csv = args.output_root / "convergence_matrix.csv"
    conv_fig = args.output_root / "convergence_heatmap.png"
    ve_csv = args.output_root / "verbosity_vs_efficiency.csv"
    ve_fig = args.output_root / "verbosity_vs_efficiency.png"

    steps = [
        (["Script 0: catalog"], [
            args.python, str(scripts_dir / "script_0_catalog.py"),
            "--run-dir", str(run_dir),
            "--csv-dir", str(csv_dir),
            "--out", str(catalog_csv),
        ]),
        (["Script 1: verbosity"], [
            args.python, str(scripts_dir / "script_1_verbosity.py"),
            "--catalog", str(catalog_csv),
            "--out-csv", str(verbosity_csv),
            "--out-fig", str(verbosity_fig),
        ]),
        (["Script 2: Task 11 metacognitive efficiency"], [
            args.python, str(scripts_dir / "script_2_task11_metacog.py"),
            "--catalog", str(catalog_csv),
            "--out-csv", str(t11_csv),
            "--out-fig", str(t11_fig),
        ]),
        (["Script 3: Task 1 delegation-by-difficulty"], [
            args.python, str(scripts_dir / "script_3_task1_delegation.py"),
            "--catalog", str(catalog_csv),
            "--out-csv", str(t1_csv),
            "--out-fig", str(t1_fig),
        ]),
        (["Script 4: Task 2 coherence"], [
            args.python, str(scripts_dir / "script_4_task2_coherence.py"),
            "--catalog", str(catalog_csv),
            "--out-csv", str(t2_csv),
            "--out-fig", str(t2_fig),
        ]),
        (["Script 5: cross-task convergence"], [
            args.python, str(scripts_dir / "script_5_convergence.py"),
            "--catalog", str(catalog_csv),
            "--verbosity-stats", str(verbosity_csv),
            "--task11-stats", str(t11_csv),
            "--task1-slopes", str(t1_slopes),
            "--task2-coherence", str(t2_csv),
            "--out-csv", str(conv_csv),
            "--out-fig", str(conv_fig),
        ]),
        (["Script 6: verbosity vs efficiency"], [
            args.python, str(scripts_dir / "script_6_verbosity_vs_efficiency.py"),
            "--convergence", str(conv_csv),
            "--out-csv", str(ve_csv),
            "--out-fig", str(ve_fig),
        ]),
    ]

    results = []
    for (name,), cmd in steps:
        ok = run_step(cmd, name)
        results.append((name, ok))

    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY")
    print("=" * 70)
    for name, ok in results:
        marker = "✓" if ok else "✗"
        print(f"  {marker}  {name}")


if __name__ == "__main__":
    main()
