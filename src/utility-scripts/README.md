# Utility scripts (figure-input builder, audit diagnostic, rank-table dump)

Three standalone Python scripts that replace the heredoc-style helpers used
in the canonical+label-grouped re-run sequence. These are reproducibility-
artifact-grade scripts: each takes CLI arguments, writes named outputs,
emits human-readable progress to stdout, and returns a meaningful exit code.

## What's in the package

```
utility-scripts/
├── README.md                              (this file)
├── build_turing_rank_reversal_input.py    (Section A in PAPER_DRAFTS)
├── diagnose_institutional_inclusion.py    (pre-flight before step 4c)
└── dump_kaggle_rank_table.py              (Kaggle per-model audit table)
```

## Where they fit in the bundle

These belong in `bundle/internal-utilities/` alongside the existing audit
scripts (`verify_three_way_overlap.py`, `generate_qc_summary.py`,
`validate_bootstrap_output.py`, `test_ece_consistency.py`). With these added,
the `internal-utilities/` folder grows to 7 scripts.

## Script-by-script

### build_turing_rank_reversal_input.py

Builds the 6-row CSV that `generate_rank_reversal_figure.py` needs as its
`--turing-csv` argument. Reads step 4b bootstrap output, drops always-
delegating models (NaN Delegation AUC), and emits a 3-column CSV
(model, ece, deleg_auc) with forced-answer ECE values.

```bash
python build_turing_rank_reversal_input.py \
    --bootstrap-csv results/reproduced/bootstrap_institutional_with_fa/bootstrap_ci_summary.csv \
    --output-csv    results/reproduced/turing_rank_reversal_input.csv
```

Defaults to requiring `declarative_source=forced_answer` for every input row;
this catches the common error of accidentally passing the wrong CSV (e.g.,
step 2 Kaggle bootstrap instead of step 4b institutional). Pass
`--no-require-forced-answer` to disable.

Exit codes: 0 on success, 1 on input-file error, 2 if fewer than 3
mixed-delegators remain (insufficient for a rank-reversal figure).

### diagnose_institutional_inclusion.py

Pre-flight diagnostic for institutional rank-divergence (step 4c).
Reports which models will actually contribute to a rank-divergence τ
estimate, and which are excluded structurally (always-delegators with
undefined Delegation AUC) or by frozen QC rule.

```bash
python diagnose_institutional_inclusion.py \
    --input-dir         results/reproduced/turing_analysis_input_80q/delegate_game/ \
    --forced-answer-dir results/results-gpqa-2026-03-25/forced_answer_phase1/ \
    --min-answered      5 \
    --output-csv        results/reproduced/institutional_inclusion_diagnostic.csv
```

Imports `metrics_for_model` and `load_forced_answer_dir` from
`compute_rank_divergence_ci.py` to use the exact computation the bootstrap
will use, so the diagnostic reflects what step 4c will actually produce.

This script is what produced the n=6 (all-attempted) and n=4 (QC-passing)
finding that motivated Option α (institutional cohort contributes per-model
results, not aggregate τ). Future bundle releases should run this script
as a standard pre-flight before any institutional rank-divergence re-run.

Issues a WARNING to stderr if the QC-passing subset is below n=6.

### dump_kaggle_rank_table.py

Builds the per-model rank-divergence audit table for the Kaggle cohort
referenced in Appendix `app:rank_divergence_audit`. Reads the Kaggle
delegate-game metrics CSV (from `analyze_kaggle_cohort.py`), filters to
models with both ECE and own-error Delegation AUC computable, and emits
a per-model table with ECE rank, AUC rank, and rank difference.

```bash
python dump_kaggle_rank_table.py \
    --input-csv  results/kaggle_cohort/comparative/delegate_game_metrics.csv \
    --output-csv results/reproduced/rank_table_canonical_labelgrouped.csv \
    --print-tau
```

The `--print-tau` flag adds Kendall τ and Spearman ρ point estimates
computed directly from the table; these should match the bootstrap point
estimates from `compute_rank_divergence_ci.py` exactly. If they don't,
something is wrong with the script wiring and step 3's bootstrap should
not be trusted.

## Why these are standalone scripts rather than heredocs

Reproducibility artifacts must be re-runnable by reviewers without
copy-pasting code from a chat log. Heredoc-style helpers are convenient
during development but are not suitable for inclusion in a release bundle.
Each of these scripts is self-contained, takes named inputs and outputs,
and produces a deterministic result given the same inputs.

The development history of how these scripts came to exist (as one-off
heredocs during the canonical-mapping + label-grouped-binning re-run
sequence) is documented in PROJECT_NARRATIVE_2026-04-25-v3.md.
