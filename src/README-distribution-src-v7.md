# MSV Benchmark — Scripts Bundle

This directory holds all analysis code. Reviewer-runnable scripts are in `distribution/`; HPC-only inference drivers are in `turing-only/`; defensive QC tools are in `internal-utilities/` and `utility-scripts/`.

## Contents at a glance

```
src/
├── README.md                       (this file)
├── distribution/                   (CPU-only, reviewer-runnable; reproduces every paper-bound number)
├── turing-only/                    (GPU + ollama; institutional inference drivers)
├── turing-cohort-analyses/         (post-hoc analyses on the 9-model institutional cohort)
├── slurm-templates/                (SLURM job templates)
├── tests/                          (consistency tests; run before trusting downstream numbers)
├── internal-utilities/             (defensive / audit utilities)
├── utility-scripts/                (audit / diagnostic one-offs)
├── kaggle-notebooks/               (11 verbatim Kaggle competition notebooks)
└── task11-notes-writeup-experiments/   (Task 11 audit pipeline + write-up notes)
```

The Makefile in the parent directory orchestrates everything; manual invocation paths are documented per-script below.

## Folder-by-folder

### `distribution/` — reviewer-runnable analysis scripts

CPU-only; runs on any laptop with `numpy/pandas/scipy/scikit-learn/matplotlib/seaborn`. This is the canonical "release with the paper" folder. Reproduces every analysis number in the paper.

| Script | Purpose | Paper section |
|---|---|---|
| `extract_kaggle_outputs.py` | Stage 1 of Kaggle pipeline. Extracts per-task CSVs from `outputs_logs_corrected.zip`. | Methods (Kaggle pipeline) |
| `adapt_kaggle_data.py` | Adapter: extractor output → bootstrap input schema (Kaggle cohort). | Methods |
| `adapt_turing_data.py` | Adapter: per-model `exp2a_delegate_trials.csv` → bootstrap input schema (Turing cohort). | Methods |
| `make_gpqa_jsonl.py` | Build `data/gpqa_diamond_80.jsonl` from GPQA Diamond CSV filtered to the Kaggle subset. | Methods (Step 4a prep) |
| `filter_to_kaggle_subset.py` | Filter Phase 2 institutional data (198 questions) to the 80-question Kaggle subset. Writes `filter_metadata.json`. | Methods (Section 5.2 cross-protocol) |
| `compute_cronbach_alpha_task4.py` | Cronbach's α for Task 4 (single-task script; preserved for backward compatibility). | Section 5.5, Limitation 10 |
| `compute_cronbach_alpha_all_tasks.py` | Cronbach's α across all 11 tasks. Reproduces Task 4 strict α = 0.978 and computes the same statistic for the remaining tasks. | Limitation 8, Limitation 10, Appendix `app:extended_results` (Cronbach's α subsection) |
| `compute_bootstrap_ci.py` | Per-model bootstrap CIs. Supports `--forced_answer_dir`. ECE uses label-grouped binning per paper Section 4.3. | All tables with CIs |
| `compute_comparative_baselines.py` | Comparative baselines: ECE, Brier, Abstention AUC, MCC, selective accuracy. Supports `--forced_answer_dir`. | Section 5.2 (Tables 3a, 3b) |
| `compute_rank_divergence_ci.py` | Bootstrap CI on rank-divergence τ. Supports `--auc_target {own_error,hardness}`, `--forced_answer_dir`. | Section 5.2, Appendix `app:rank_divergence_audit` |
| `analyze_kaggle_cohort.py` | Cross-cohort analysis on 23-model Kaggle output. Powers Sections 5.4, 5.5, App C. | Sections 5.4, 5.5 (Kaggle cohort) |
| `compute_task10_dpp_analysis.py` | Per-model matched-paired lift, win/loss decomposition, McNemar exact, Expert→Generalist trace correction analysis. | Appendix `app:task10_dpp_institutional` |
| `compute_task10_mde.py` | Minimum detectable effect (MDE) per model for the Task 10 matched-lift analysis. Defends the negative result by quantifying detectability. | Limitation 1, Limitation 11, Appendix `app:extended_results` (MDE subsection) |
| `generate_completion_heatmap.py` | Figure 6: completion rates across models × tasks. | Appendix D |
| `generate_msv_routing_subset_analysis.py` | Figure 5: MSV dimension-subset routing. | Appendix D |
| `generate_rank_reversal_figure.py` | Figure 1 (top): behavioral-vs-declarative rank-reversal scatter. | Section 5.2 |
| `generate_ece_vs_delegauc_scatter.py` | Figure 1 (bottom): raw-value scatter of ECE vs Delegation AUC-ROC for the 11-model Kaggle subset, companion to the rank panels. | Section 5.2, `fig:ece_vs_delegauc` |
| `generate_risk_coverage_figure.py` | Figure 2: risk-coverage curves on 4 Kaggle models. | Section 5.2 |

### `turing-only/` — GPU + ollama required

Requires the institutional HPC cluster with GPU + ollama (via Apptainer SIF). Reviewers without similar infrastructure cannot run these directly, but the **outputs** are bundled at `data/forced_answer_phase1/` and `data/task10_dpp/` so all downstream analysis runs without re-collecting the data.

| Script | Purpose | Paper section |
|---|---|---|
| `run_forced_answer_phase1_turing.py` | Forced-answer Phase 1 on the institutional 9-model cohort. | Section 5.2 (Table 3b); Limitation 6 |
| `run_task10_dpp_turing.py` | Task 10 DPP run on the institutional cohort at `NUM_CTX=32768`. | Appendix `app:task10_dpp_institutional`; Limitation 11 |

For the Turing-specific operational setup (Apptainer `--userns` flag, `${HOME}/.ollama/models` bind path, model pulls, SLURM templates), see `REPRODUCIBILITY_GUIDE.md` Section 6.

### `turing-cohort-analyses/` — institutional cohort post-hoc analyses

| Script | Purpose |
|---|---|
| `compute_ranking_stability.py` | Split-half ranking stability on the institutional cohort. |
| `compute_sensitivity_analysis.py` | Reward-schedule and difficulty-threshold perturbations. |
| `compute_synthetic_delegation.py` | Synthetic delegation comparison. |

### `slurm-templates/` — SLURM job templates

| Template | Job |
|---|---|
| `slurm_bootstrap_ci.sh` | Per-model bootstrap CIs on Turing |
| `slurm_rank_divergence_ci.sh` | Rank-divergence bootstrap on Turing |
| `slurm_run_forced_answer.sh` | Forced-Answer Phase 1 inference |
| `slurm_run_task10_dpp.sh` | Task 10 DPP inference (12-hour wall-clock cap, 24 GB RAM) |

### `tests/` — consistency tests (run first)

| Test | What it checks |
|---|---|
| `test_ece_consistency.py` | All four ECE-computing scripts (`compute_bootstrap_ci.py`, `compute_rank_divergence_ci.py`, `analyze_kaggle_cohort.py`, `compute_comparative_baselines.py`) produce identical output on a fixed test panel. Catches binning regressions. |
| `test_hardness_auc_consistency.py` | `compute_delegation_auc_vs_hardness` matches `sklearn.metrics.roc_auc_score`; `attach_global_hardness_label` produces stable question-level labels. Catches hardness-target drift. |

Both tests must report "ALL TESTS PASSED" before downstream numbers should be trusted. Run via `make tests`.

### `internal-utilities/` — defensive / audit utilities

Not invoked by the headline pipeline, but encode the methodological discipline that caught real failures during pipeline development.

| Script | Purpose |
|---|---|
| `verify_three_way_overlap.py` | Pre-flight check before institutional bootstrap. Confirms `Phase2 ∩ forced-answer ∩ Kaggle-subset = 80` for every model. Exits non-zero on partial overlap. |
| `generate_qc_summary.py` | Generates `qc_summary.csv` from forced-answer CSVs by applying QC rules (hard QC: `valid >= 70/80, parse_fail <= 10/80`; soft flags for answer-anchor / confidence-collapse / low-conf-scale-use). |
| `validate_bootstrap_output.py` | Post-bootstrap validation. Checks row count, `declarative_source` distribution, CI containment, optional model-set membership. |

### `utility-scripts/` — diagnostic one-offs

| Script | Purpose |
|---|---|
| `build_turing_rank_reversal_input.py` | Build the input table for the institutional panel of Figure 1. |
| `diagnose_institutional_inclusion.py` | Diagnose why a particular model was/wasn't included in a comparative table. |
| `dump_kaggle_rank_table.py` | Per-model Kaggle rank table (audit aid). |

### `kaggle-notebooks/` — Kaggle competition notebooks

11 verbatim competition notebooks, one per task (`t01-msv-delegate-game.ipynb` through `t11-msv-mc-binary-pairs.ipynb`). Prompt text is also transcribed in Appendix F of the paper.

### `task11-notes-writeup-experiments/` — Task 11 audit pipeline

Two parallel script directories:

- `src-msv-analysis/` — canonical release copy with `.orig` baselines for forensic comparison
- `task11_analysis/scripts/` — live workspace mirror

Both directories contain the same 7-script pipeline orchestrated by `run_all.py`:

| Script | Output |
|---|---|
| `script_0_catalog.py` | Run catalog: which (model, task) runs are available |
| `script_1_verbosity.py` | Per-model verbosity index (mean output tokens/response) |
| `script_2_task11_metacog.py` | Per-model Type-2 AUC, MC, $\hat{d}$ on Task 11 with raw-confidence recovery |
| `script_3_task1_delegation.py` | Task 1 delegation curves (rate vs difficulty) |
| `script_4_task2_coherence.py` | Task 2 declared-routing coherence (Spearman ρ per model) |
| `script_5_convergence.py` | Cross-task convergence matrix → `convergence_matrix.{png,csv}` |
| `script_6_verbosity_vs_efficiency.py` | Verbosity vs MC scatter (3-panel small-multiples) |

Reproduces Appendix `app:task11_audit` plus the two paper figures (`d_hat_vs_type2auc.png` from script 2 and `convergence_matrix.png` from script 5).

The pipeline reads from `data/kaggle-data/kaggle_extracted/per_task/` and `kaggle_raw/outputs_logs_corrected.zip` (extracted to a working directory at run-time). Outputs go to `task11_analysis/outputs/`; the Makefile copies the canonical CSVs to `results/task11_audit/` and the figures to `results/figures/`.

The five `*.orig` files in `src-msv-analysis/` are the pre-audit baseline (for forensic comparison; referenced from project `RUN_LOG_SUPPLEMENTAL.md` modifications #6-#11). They are intentional and should not be deleted.

## Pipeline invariants

Four numerical conventions are encoded in the code and verified by the tests in `tests/`:

1. **Confidence-to-probability mapping:** `{1, 2, 3, 4} → {0.25, 0.50, 0.75, 1.00}` (paper Section 4.3). Enforced in: `compute_bootstrap_ci.py`, `compute_rank_divergence_ci.py`, `analyze_kaggle_cohort.py`, `compute_comparative_baselines.py`.

2. **ECE binning:** label-grouped (one bin per discrete confidence level), not equal-width over [0,1] (paper Section 4.3). Verified by `tests/test_ece_consistency.py`.

3. **Hardness AUC:** uses the global panel median, not per-model medians, for paper-bound results (paper Appendix `app:rank_divergence_audit`). Verified by `tests/test_hardness_auc_consistency.py`. The per-model-median variant is preserved in the codebase as `deleg_auc_vs_hardness_per_model_median` for backward compatibility but is NOT cited anywhere in the paper.

4. **Bootstrap CIs:** 10,000 question-panel resamples for paper-bound numbers. Default `--n_boot 10000`. SLURM templates set `N_BOOT=10000`. Lower values are acceptable for development but paper-bound numbers all use 10,000.

If a reproduction produces different numbers, re-run the consistency tests first. If those pass, check whether one of the four invariants has been violated by a local modification.

## Re-run instructions

The `Makefile` in the parent directory orchestrates everything:

```bash
cd ..             # back to distribution-msv-benchmark/

# Tests (run first; must pass)
make tests

# Headline analyses
make reproduce-stats           # Cronbach α (Task 4 + all-tasks) + bootstrap CIs + rank divergence
make reproduce-figures         # all 6 paper figures
make reproduce-task10-dpp      # Task 10 DPP appendix analysis + MDE
make reproduce-task11          # Task 11 audit appendix analysis

# Or all of the above:
make reproduce-all
```

Outputs land in `results/reproduced/` so the bundled `results/` stays authoritative. Compare bit-for-bit between the two for verification.
