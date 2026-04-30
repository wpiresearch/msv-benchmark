# MSV Benchmark — Reproducibility Guide

This guide documents how to reproduce every paper-bound number from the bundled artifact, what is bundled vs. what is not, and how to verify that a reproduction matches the released values.

## Contents

1. [Quick start](#1-quick-start)
2. [What is reproducible from this package alone (no HPC needed)](#2-what-is-reproducible-from-this-package-alone-no-hpc-needed)
3. [Data: what's bundled, what's not](#3-data-whats-bundled-whats-not)
4. [Pipeline invariants — read this before running anything](#4-pipeline-invariants--read-this-before-running-anything)
5. [Step-by-step reproduction (Kaggle cohort, no HPC needed)](#5-step-by-step-reproduction-kaggle-cohort-no-hpc-needed)
6. [Step-by-step reproduction (institutional cohort, requires HPC)](#6-step-by-step-reproduction-institutional-cohort-requires-hpc)
7. [Per-claim provenance](#7-per-claim-provenance)
8. [Figure regeneration](#8-figure-regeneration)
9. [Psychometric reliability and detectability analyses](#9-psychometric-reliability-and-detectability-analyses)
10. [Computational requirements](#10-computational-requirements)
11. [Verification](#11-verification)
12. [Cross-references](#12-cross-references)

---

## 1. Quick start

```bash
cd distribution-msv-benchmark/
make setup                  # install Python dependencies
make tests                  # run consistency tests (must pass)
make reproduce-all          # run everything (~45 min sequential at full bootstrap)
```

The result is a populated `results/reproduced/` directory whose contents bit-for-bit match the bundled `results/` (modulo small bootstrap-resampling variance in the third decimal of CI bounds).

For development with smaller bootstrap iteration counts:

```bash
make reproduce-stats BOOT=500   # ~5 minutes instead of ~30
```

---

## 2. What is reproducible from this package alone (no HPC needed)

**Yes (CPU-only, ~45 minutes total):**

- All Kaggle-cohort analyses including the τ = +0.45 cross-cohort headline result
- Cronbach's α across the full 11-task suite (Section 9 of this guide; Appendix `app:extended_results`)
- Per-model bootstrap CIs on declarative and behavioral metrics
- Rank-divergence audit at all inclusion thresholds (own-error AUC primary; hardness AUC alternative)
- All 6 paper figures
- Task 11 audit pipeline including the Haiku 4.5 raw-confidence recovery and the cross-task convergence matrix
- Task 10 DPP institutional analysis (matched-paired lift, win/loss, McNemar, Expert→Generalist trace, MDE)

**No (requires HPC re-run):**

- Re-running the actual model inference for the institutional 9-model cohort (Forced-Answer Phase 1, Task 10 DPP). The outputs of those runs are bundled at `data/forced_answer_phase1/` and `data/task10_dpp/`, so all downstream analysis runs without the inference step.
- Re-running the 23 Kaggle-platform models. The Kaggle competition outputs are bundled as `data/kaggle-data/kaggle_raw/outputs_logs_corrected.zip` and pre-extracted into `data/kaggle-data/kaggle_extracted/per_task/`.

If you have HPC + ollama + the relevant model weights, Section 6 of this guide describes the inference re-runs. Otherwise, Section 5 is the canonical reproduction path.

---

## 3. Data: what's bundled, what's not

### 3.1 GPQA Diamond panel

The GPQA Diamond dataset is publicly available from Rein et al. (2023) and **is not redistributed** in this package. We provide:

- `data/gpqa_difficulty_scores.csv` — 198 question IDs paired with empirical difficulty labels (the fraction of our 10-model calibration cohort that answered incorrectly)
- The 80-question Kaggle subset is specified by the question_ids appearing in `data/kaggle-data/kaggle_extracted/per_task/t01_delegate_game.csv`

To retrieve the question text, clone https://github.com/idavidrein/gpqa and filter to those IDs. The reproduction pipeline below does not require the question text because the per-trial model outputs (which include the model's response only, not the question) are bundled directly.

### 3.2 Kaggle competition raw outputs

`data/kaggle-data/kaggle_raw/outputs_logs_corrected.zip` (16 MB; sha256 `21c67de1...77d4f`) holds the 253 task-model archives produced by the Kaggle Benchmarks platform during the competition run. The `extract_kaggle_outputs.py` extractor in `src/distribution/` turns this into:

- `data/kaggle-data/kaggle_extracted/per_task/t??_*.csv` — 11 long-form per-trial CSVs
- `data/kaggle-data/kaggle_extracted/per_model/<model>/*.csv` — 23 model directories
- `data/kaggle-data/kaggle_extracted/run_metadata.csv` — 253-row task-model registry

The pre-extracted CSVs are bundled, so the extractor is needed only for end-to-end reproduction starting from the raw archive.

### 3.3 Croissant 1.0 metadata

`data/kaggle-data/croissant_metadata.json` is a hand-authored Croissant 1.0 metadata card describing the per-trial outputs. It includes:

- All RAI fields required by NeurIPS (license, terms, intended use, limitations)
- File-level sha256 hashes for the raw archive and extracted CSVs (verified to match)
- Field-level descriptions for each per-task CSV
- Pointer to the GPQA upstream repo

A Croissant 1.0 validator can be invoked via `make validate-croissant`.

### 3.4 What's NOT bundled

- GPQA Diamond question text (see Section 3.1)
- Per-trial outputs from the institutional HPC runs older than the Forced-Answer Phase 1 and Task 10 DPP collections (the Phase 2 198-question Delegate Game institutional runs from earlier project phases). The summary tables and analysis-ready CSVs derived from them are bundled where they are referenced in the paper. The driver scripts allow re-collection if needed.
- Author-identifying metadata (the package is anonymized for double-blind submission). After acceptance the public release will include the author-identifying Croissant fields.

---

## 4. Pipeline invariants — read this before running anything

These four invariants are encoded in code and verified by `src/tests/test_*.py`. If a reproduction produces different numbers, the first thing to check is whether one of these invariants has been violated by a local modification.

1. **Confidence-to-probability mapping is canonical:** `{1, 2, 3, 4} → {0.25, 0.50, 0.75, 1.00}`. Documented in paper Section 4.3. Enforced in `compute_bootstrap_ci.py`, `compute_rank_divergence_ci.py`, `analyze_kaggle_cohort.py`, `compute_comparative_baselines.py`. Any other mapping (linear endpoints `{0, 1/3, 2/3, 1}`) will produce different ECE/Brier values.

2. **ECE binning is label-grouped (one bin per discrete confidence level), NOT equal-width over [0,1].** Documented in paper Section 4.3. Verified by `src/tests/test_ece_consistency.py`. Equal-width binning silently merges confidence levels 3 and 4 into the bin `[0.75, 1.00]` and leaves `[0.00, 0.25)` empty under the canonical mapping; this gives systematically different ECE values.

3. **Hardness-AUC uses the global panel median, not per-model medians, for paper-bound results.** Documented in paper Appendix `app:rank_divergence_audit`. Verified by `src/tests/test_hardness_auc_consistency.py`. The per-model-median variant is preserved in the codebase as `deleg_auc_vs_hardness_per_model_median` for backward compatibility but is not cited anywhere in the paper.

4. **Bootstrap CIs use 10,000 question-panel resamples.** Default `--n_boot 10000` in all bootstrap scripts. SLURM templates set `N_BOOT=10000`. Lower values are acceptable for development but paper-bound results all use 10,000.

Run the consistency tests first:

```bash
make tests
```

Both should report "ALL TESTS PASSED" and exit 0. If either fails, the local copy of the scripts has drifted from the canonical implementation; do not trust downstream numbers.

---

## 5. Step-by-step reproduction (Kaggle cohort, no HPC needed)

This sequence reproduces every paper-bound number from `data/kaggle-data/kaggle_extracted/`. Total wall-clock: ~2.5 hours sequential at full bootstrap iterations, or ~5 minutes with `BOOT=500` for development.

| Step | Make target | Wall-clock | Output |
|---|---|---:|---|
| 1 | `make tests` | 1 min | "ALL TESTS PASSED" on both consistency tests |
| 2 | `make reproduce-stats` | 30 min full / 5 min BOOT=500 | Cronbach α (Task 4 + all-tasks), per-model bootstrap CIs, rank divergence (own-error + hardness) |
| 3 | `make reproduce-figures` | 5 min | Figures 1, 2, 5, 6 (4 of 6) |
| 4 | `make reproduce-task11` | 10 min | Figures 3, 4 + Task 11 audit CSVs |
| 5 | `make reproduce-task10-dpp` | 5 min | Task 10 DPP appendix tables (lift, win/loss, trace, MDE) |

For end-to-end Kaggle pipeline starting from `kaggle_raw/outputs_logs_corrected.zip`:

```bash
make reproduce-kaggle      # adds 2 more minutes for the extraction stage
```

**Verification:** Each output has expected point estimates documented in this guide and Section 11 below. A reviewer can compare bit-for-bit against the bundled `results/`.

---

## 6. Step-by-step reproduction (institutional cohort, requires HPC)

The institutional 9-model cohort runs ollama under apptainer on a single GPU per model. This section is needed only if you want to re-run the actual model inference (not just re-analyze the bundled outputs). For analysis-only reproduction, Section 5 plus the bundled `data/forced_answer_phase1/` and `data/task10_dpp/` directories are sufficient.

### 6.1 Environment setup

The institutional pipeline uses Apptainer (formerly Singularity) with an ollama SIF image. The relevant operational details:

- Apptainer must be invoked with `--userns` to allow the ollama daemon to bind-mount the user-writable model store
- Model weights are stored at `${HOME}/.ollama/models` and bound into the container at `/root/.ollama/models`
- The 9 models pulled for the cohort: `qwen2.5:7b`, `qwen2.5:3b`, `llama3.1:8b`, `llama3.2:3b`, `llama3.2:1b`, `phi4-mini`, `gemma2:9b`, `gemma2:2b`, `mistral:7b`
- SLURM templates in `src/slurm-templates/` assume a 12-hour wall-clock cap and 24 GB system RAM per job

### 6.2 Forced-Answer Phase 1 collection

Forced-Answer Phase 1 elicits an answer + 1-4 confidence rating on every one of the 80-question Kaggle subset for each of the 9 institutional models, independently of any delegation policy. This is the calibration-repair protocol referenced in Section 5.2 (Table 3) and Limitation 6.

```bash
# Submit one SLURM job per model
for model in qwen2.5:7b qwen2.5:3b llama3.1:8b llama3.2:3b llama3.2:1b \
             phi4-mini gemma2:9b gemma2:2b mistral:7b; do
    sbatch src/slurm-templates/slurm_run_forced_answer.sh "$model"
done
```

Wall-clock: ~2 hours per model on a single GPU; ~50 minutes total cohort wall-clock under a 4-GPU per-user QoS limit. Outputs land in `results/reproduced/forced_answer_phase1/<model>.csv`.

### 6.3 Task 10 DPP institutional rerun

The Task 10 DPP institutional rerun executes the five-stage Expert→Critic→Evaluator→Synthesizer→Generalist pipeline at `NUM_CTX=32768` on the same 80-question subset for each of the 9 models. Reported in Appendix `app:task10_dpp_institutional` and Limitation 11.

```bash
# Submit one SLURM job per model
for model in qwen2.5:7b qwen2.5:3b llama3.1:8b llama3.2:3b llama3.2:1b \
             phi4-mini gemma2:9b gemma2:2b mistral:7b; do
    sbatch src/slurm-templates/slurm_run_task10_dpp.sh "$model" 32768
done
```

Wall-clock: 9-21 minutes per model; ~50 minutes cohort wall-clock under a 4-GPU per-user QoS limit. Outputs land in `results/reproduced/task10_dpp/<model>.csv` plus `results/reproduced/task10_dpp/<model>__transcripts/<question_id>.json`.

The DPP analysis stage (matched-lift, win/loss, trace correction, MDE) is CPU-only and is invoked by `make reproduce-task10-dpp`. See Section 9 of this guide.

### 6.4 Task 11 audit pipeline

The Task 11 audit pipeline processes the bundled Kaggle outputs (no inference re-run needed) to produce the cross-task convergence matrix and the corrected Haiku 4.5 metacognitive efficiency. Run via:

```bash
make reproduce-task11
```

This invokes `src/task11-notes-writeup-experiments/src-msv-analysis/run_all.py`, which orchestrates the 7-script pipeline. The outputs include the two main-paper figures (`d_hat_vs_type2auc.png` from script 2 and `convergence_matrix.png` from script 5) and the canonical CSVs for `results/task11_audit/`.

---

## 7. Per-claim provenance

This section maps every quantitative claim in the paper to the script + bundled CSV + paper section that produces it.

### 7.1 Section 1 (Introduction)

- llama3.2:1b answered-conditional ECE = 0.42 on 142/198, Delegation AUC = 0.60 → `results/kaggle_cohort/comparative/delegate_game_metrics.csv` row `llama3.2:1b`
- llama3.1:8b answered-conditional ECE = 0.58 on 56/198, Delegation AUC = 0.90 → same file, row `llama3.1:8b`

### 7.2 Section 3.2 (Dataset Construction)

- Empirical difficulty labels for the 80-question subset → `data/gpqa_difficulty_scores.csv`

### 7.3 Section 4 (Experimental Setup)

- 9-model institutional cohort enumerated → `data/forced_answer_phase1/` and `data/task10_dpp/` directories
- 23-model Kaggle cohort enumerated → `data/kaggle-data/kaggle_extracted/run_metadata.csv` `model` column

### 7.4 Section 5.1 (Behavioral Metacognitive Control)

- Table 1 (Delegate Game results) → `results/turing_cohort/exp2a_results_summary.csv`

### 7.5 Section 5.2 (the central analysis)

- Table 2 (same-output answered-conditional, institutional) → `results/turing_cohort/comparative_table_aconcond.csv`
- Table 3 (same-item cross-protocol forced-answer) → `results/turing_cohort/comparative_table_fa.csv`
- Cross-cohort Kendall τ = +0.45 (95% CI [+0.02, +0.62]) → `results/kaggle_cohort/rank_divergence_bootstrap.csv` row `kaggle_cohort_aggregate` for own-error AUC at min_answered=5
- Spearman ρ = +0.64 (95% CI [+0.03, +0.80]) → same file
- Maximum point rank gap Δ=5 (claude-opus-4-6-default) → same file
- Figure 1, top panels (rank-reversal scatter) → `results/figures/rank_reversal_scatter.{png,pdf}`
- Figure 1, bottom panel (ECE vs. Delegation AUC raw-value scatter, `fig:ece_vs_delegauc`) → `results/figures/ece_vs_delegauc_scatter.{png,pdf,csv}`
- Figure 2 (risk-coverage curves) → `results/figures/risk_coverage.{png,pdf}`

### 7.6 Section 5.3 / Appendix C

- Companion task headlines (Table 4) → `results/kaggle_cohort/companion_task_headlines.csv`
- Per-model companion-task scores → `results/kaggle_cohort/per_model_task_scores.csv`

### 7.7 Section 5.4 (Sensitivity Analysis)

- 17 perturbations + 1000 splits → `results/kaggle_cohort/sensitivity/`

### 7.8 Section 5.5 — Cronbach α

- Per-task α (Task 4 strict α = 0.978) → `results/psychometrics/cronbach_alpha_task4.txt`
- Full all-tasks α table (Limitation 8 evidence: Task 7 = 0.965, Task 8 = -1.571 etc.) → `results/psychometrics/cronbach_alpha_all_tasks.{csv,txt}`
- See Section 9 of this guide for the all-tasks computation details

### 7.9 Section 5.5 — Cross-Task Dissociations

- 20 of 23 models with rank range ≥ 13 → `results/kaggle_cohort/dissociations/per_model_profile.csv`
- Cross-task Spearman matrix → `results/kaggle_cohort/dissociations/cross_task_spearman.csv`
- MSV routing subset finding (CE-only worst single-dimension predictor at ρ = -0.315) → `results/figures/msv_routing_subset_analysis.csv`

### 7.10 Section 6 (Limitations)

- Limitation 1 (MDE for matched-lift) → `results/task10_dpp/task10_mde.csv` (cohort pooled MDE = 0.046; per-model 0.110-0.190)
- Limitation 8 (Task 6/7 high α; Task 8 negative α) → `results/psychometrics/cronbach_alpha_all_tasks.csv`
- Limitation 10 (per-task α reliability tiers) → same file
- Limitation 11 (Task 10 DPP infrastructure-feasibility-but-not-efficacy + MDE) → `results/task10_dpp/task10_lift.csv` + `task10_mde.csv`
- Limitation 12 (Task 11 five failure modes) → `results/task11_audit/task11_metacognitive_efficiency.csv`

### 7.11 Appendix `app:task11_audit`

- Five failure modes (preprocessing collapse, scale collapse, parsing failure, anti-calibration, verbose-CoT) → `results/task11_audit/task11_metacognitive_efficiency.csv` + `convergence_matrix.csv`
- Haiku 4.5 corrected Type-2 AUC = 0.747 (95% CI [0.668, 0.818]) → `task11_metacognitive_efficiency.csv` row `claude-haiku-4-5-20251001`
- Cross-task convergence matrix figure → `results/figures/convergence_matrix.{png,pdf}`

### 7.12 Appendix `app:task10_dpp_institutional`

- Table `tab:task10_lift` (per-model end-to-end and matched lifts with 95% CIs) → `results/task10_dpp/task10_lift.csv`
- Table `tab:task10_winloss` (paired wins/losses, churn, McNemar p) → `results/task10_dpp/task10_winloss.csv`
- Table `tab:task10_trace` (Expert→Generalist correction analysis) → `results/task10_dpp/task10_trace.csv`
- Table `tab:task10_mde` (per-model minimum detectable effect) → `results/task10_dpp/task10_mde.csv`
- Cohort-aggregate numbers (120 wins, 109 losses, 229 discordant, 648 matched, 0.35 churn) → `task10_winloss.csv` row `COHORT_AGGREGATE`
- llama3.1:8b net +9 rescues, qwen2.5:3b net -5 harms, two-sided binomial p ≈ 0.093 / 0.063 → `task10_trace.csv`

---

## 8. Figure regeneration

All 7 paper figures regenerate from bundled CSVs in under 15 minutes total. (Figure 1 has a top row of two rank-reversal panels and a bottom row holding the new ECE-vs-Delegation-AUC scatter; the three Figure-1 sub-panels share a single floating environment in the paper but are produced by two separate scripts.) Each has a single canonical generation script:

| Figure | Section | Script | Inputs |
|---|---|---|---|
| Figure 1, top panels: Rank-reversal scatter | 5.2 | `src/distribution/generate_rank_reversal_figure.py` | `results/kaggle_cohort/comparative/delegate_game_metrics.csv` |
| Figure 1, bottom panel: ECE vs. Delegation AUC raw-value scatter | 5.2 | `src/distribution/generate_ece_vs_delegauc_scatter.py` | `results/kaggle_cohort/comparative/delegate_game_metrics.csv` |
| Figure 2: Risk-coverage curves | 5.2 | `src/distribution/generate_risk_coverage_figure.py` | `data/kaggle-data/kaggle_extracted/per_task/t01_delegate_game.csv` |
| Figure 3: Type-2 AUC vs $\hat{d}$ scatter | App `app:task11_audit` | `src/task11-notes-writeup-experiments/src-msv-analysis/script_2_task11_metacog.py` (via `run_all.py`) | Kaggle Task 11 CSVs |
| Figure 4: Cross-task convergence heatmap | App `app:task11_audit` | `src/task11-notes-writeup-experiments/src-msv-analysis/script_5_convergence.py` (via `run_all.py`) | Outputs of scripts 1-4 |
| Figure 5: MSV routing subset | App D | `src/distribution/generate_msv_routing_subset_analysis.py` | `data/kaggle-data/kaggle_extracted/per_task/t02_declared_probe.csv`, `run_metadata.csv` |
| Figure 6: Completion heatmap | App D | `src/distribution/generate_completion_heatmap.py` | `data/kaggle-data/kaggle_extracted/run_metadata.csv` |

Run with:

```bash
make reproduce-figures      # Figures 1, 2, 5, 6 (4 figures)
make reproduce-task11       # Figures 3, 4 (the Task 11 audit pair)
```

Both targets together regenerate all 7 figures (the 3 sub-panels of Figure 1 plus Figures 2-6). Compare bit-for-bit with the bundled `results/figures/`.

---

## 9. Psychometric reliability and detectability analyses

Two analyses report psychometric and methodological details that support the paper's Limitations and the per-task reliability claims in Appendix `app:extended_results`.

### 9.1 Cronbach's α across all tasks

`compute_cronbach_alpha_all_tasks.py` generalizes the existing `compute_cronbach_alpha_task4.py` to all 11 tasks. It produces the per-task α table at three balance conventions (strict t=23, relaxed t=20, inclusive t=15) and per-domain stratification (GPQA categories) where the item key is the GPQA `question_id`.

```bash
python src/distribution/compute_cronbach_alpha_all_tasks.py \
    --per-task-dir data/kaggle-data/kaggle_extracted/per_task/ \
    --output-prefix results/reproduced/cronbach_alpha_all_tasks
```

The script reproduces the existing Task 4 strict α = 0.978 exactly (validation that the convention matches the existing Task 4 script). For Tasks 1, 3, 5, 6, 7, 11 it computes α at all three thresholds; for Task 2 it uses the `routing_score` column; for Task 8 (indexed by pair ID rather than `question_id`) and Tasks 9, 10 (low completion) the result is pooled-only or interpretively limited.

**Headline values (paper Limitation 8 and 10 reference these):**

| Task | α (strict) | α (t=15) | Reliability tier |
|---|---|---|---|
| Task 1 (Delegate Game)      | 0.575 | 0.875 | acceptable |
| Task 2 (Declared MSV)       | 0.974 | 0.989 | excellent |
| Task 3 (Second-Chance)      | 0.666 | 0.879 | acceptable |
| Task 4 (Confidence Entropy) | 0.978 | 0.976 | excellent |
| Task 5 (Teammate Delegate)  | 0.755 | 0.924 | good |
| Task 6 (Behavioral ER)      | 0.723 | 0.958 | good |
| Task 7 (Behavioral CI)      | 0.965 | 0.994 | excellent |
| **Task 8 (Behavioral EM)**  | **−0.000** | **−1.571** | **construct-degenerate** |
| Task 9 (Behavioral PI)      | n/a   | 0.995 | excellent (when defined) |
| Task 10 (DPP)               | 0.365 | 0.018 | uninterpretable (low k) |
| Task 11 (MC Binary Pairs)   | 0.530 | 0.869 | moderate to good |

The Task 8 negative α at every threshold is a substantive finding rather than a power issue: items measuring the EM construct produce inversely correlated responses across models, consistent with the proxy-validity caveat reported in Section 3.3 (Tasks 6-10 paragraph) and strengthening Limitation 8.

Outputs land at `results/psychometrics/cronbach_alpha_all_tasks.{csv,txt}`.

### 9.2 Minimum detectable effect for Task 10 matched-lift

`compute_task10_mde.py` quantifies what magnitude of per-model lift the 80-item panel could have detected at α = 0.05 two-sided, given each model's empirical paired-diff variability. Reported in Limitation 11 and Appendix `app:extended_results`.

```bash
python src/distribution/compute_task10_mde.py \
    --dpp-dir   data/task10_dpp/ \
    --fa-dir    data/forced_answer_phase1/ \
    --output-csv results/reproduced/task10_mde.csv
```

For each model, MDE = z_{0.975} · s / √n_matched, where s is the SD of the paired-diff vector and n_matched is the FA-parsed inner-join count.

**Headline values:**

| Model | n_matched | observed lift | s | MDE α=0.05 two-sided | observed below MDE |
|---|---|---|---|---|---|
| gemma2:2b   | 80 | +0.113 | 0.503 | 0.110 | **no** (at threshold) |
| gemma2:9b   | 78 | +0.000 | 0.581 | 0.129 | yes |
| llama3.1:8b | 74 | +0.122 | 0.572 | 0.130 | yes |
| llama3.2:1b | 64 | +0.016 | 0.549 | 0.135 | yes |
| llama3.2:3b | 73 | -0.027 | 0.666 | 0.153 | yes |
| mistral:7b  | 77 | -0.078 | 0.602 | 0.134 | yes |
| phi4-mini   | 48 | +0.125 | 0.672 | 0.190 | yes |
| qwen2.5:3b  | 77 | -0.065 | 0.522 | 0.117 | yes |
| qwen2.5:7b  | 77 | -0.013 | 0.679 | 0.151 | yes |
| **Cohort pooled** | 648 | +0.017 | 0.595 | **0.046** | yes |

Per-model MDE ranges from 0.110 to 0.190; cohort-pooled MDE is 0.046. Eight of nine models had observed |lift| < MDE; gemma2:2b sits at the threshold (consistent with its near-significant McNemar p = 0.078). The MDE complements the bootstrap-CI result by showing that the 80-item panel had power to detect lifts of ~13 percentage points per-model and ~5 percentage points pooled, but the observed effects fell below those thresholds. We interpret this as a panel-power-bounded null rather than evidence of a non-zero effect that the panel was too small to find.

Outputs land at `results/task10_dpp/task10_mde.csv` (alongside `task10_lift.csv`, `task10_winloss.csv`, `task10_trace.csv`).

---

## 10. Computational requirements

| Task | CPU | RAM | GPU | Wall-clock |
|---|---|---|---|---|
| `make tests` | 1 core | <500 MB | none | 1 min |
| `make reproduce-stats` (BOOT=10000) | 1 core | <2 GB | none | ~30 min |
| `make reproduce-stats` (BOOT=500) | 1 core | <2 GB | none | ~5 min |
| `make reproduce-figures` | 1 core | <500 MB | none | ~5 min |
| `make reproduce-task10-dpp` | 1 core | <500 MB | none | ~10 sec |
| `make reproduce-task11` | 1 core | <2 GB | none | ~10 min |
| Forced-Answer Phase 1 (per model) | 1 core | 24 GB | 1 GPU | ~2 hours |
| Task 10 DPP rerun (per model, NUM_CTX=32768) | 1 core | 24 GB | 1 GPU | 9-21 min |

For analysis-only reproduction, a single-core laptop with 4 GB RAM and Python 3.10+ is sufficient. The HPC re-runs are needed only for re-collecting model outputs.

---

## 11. Verification

After reproducing any number, compare against the bundled `results/` and the locked values documented in this guide.

**Headline values to check:**

| Quantity | Value | Source |
|---|---|---|
| Cross-cohort Kendall τ | +0.4545, 95% CI [+0.022, +0.624] | `rank_divergence_bootstrap.csv` row `kaggle_cohort_aggregate` |
| Cross-cohort Spearman ρ | +0.6364, 95% CI [+0.030, +0.800] | same file |
| Cronbach α (Task 4 strict) | 0.978 | `cronbach_alpha_task4.txt` and `cronbach_alpha_all_tasks.csv` (validation cross-check) |
| Cronbach α (Task 7 strict) | 0.965 | `cronbach_alpha_all_tasks.csv` |
| Cronbach α (Task 8 strict) | −0.000 | `cronbach_alpha_all_tasks.csv` |
| Cronbach α (Task 8 inclusive t=15) | −1.571 | same file (construct-degenerate finding) |
| Haiku 4.5 Type-2 AUC (corrected) | 0.747, 95% CI [0.668, 0.818] | `task11_audit/task11_metacognitive_efficiency.csv` |
| Task 10 DPP cohort-mean matched lift | +0.021, range [-0.078, +0.125] | `task10_dpp/task10_lift.csv` |
| Task 10 DPP cohort-aggregate wins/losses | 120 / 109 / 229 discordant / 648 matched / 0.35 churn | `task10_dpp/task10_winloss.csv` |
| Task 10 cohort-pooled MDE (α=0.05 two-sided) | 0.046 | `task10_dpp/task10_mde.csv` row `COHORT_POOLED` |
| ECE vs. Delegation AUC scatter, n_plottable | 11 of 23 Kaggle models | `results/figures/ece_vs_delegauc_scatter.csv` (rows where both `ece` and `deleg_auc` are non-null) |
| Task 10 llama3.1:8b binomial p (rescues vs harms, two-sided) | 0.0931 | `task10_dpp/task10_trace.csv` |

Bit-for-bit match is expected up to 16 decimal places for point estimates and 3 decimal places for CIs (small bootstrap variance).

If a number doesn't match: first re-run `make tests`. If those pass, check the four pipeline invariants in Section 4. If those check out, the discrepancy may correspond to a known modification; the project's audit history is preserved in `docs-notes/RUN_LOG_SUPPLEMENTAL.md` (project-internal, not in this package).

---

## 12. Cross-references

- `README.md`: package overview and quick-start
- `data/README.md`: data provenance, schemas, license
- `src/README.md`: scripts bundle overview
- `Makefile`: orchestrates all reproductions
- `requirements.txt`: pinned Python dependencies
- `LICENSE`: MIT for code; CC-BY-4.0 for derived metadata
