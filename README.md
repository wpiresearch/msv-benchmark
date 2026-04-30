# MSV Benchmark — Reproducibility Package

This repository accompanies the submission to the NeurIPS 2026 Evaluations & Datasets track:

> **Declarative vs. Behavioral Metacognition in Large Language Models: A Benchmark on Expert Questions**

It contains the data, code, and pre-computed results needed to reproduce every quantitative claim in the paper. The package is self-contained: no HPC access required for any analysis or figure regeneration. Re-running model inference (forced-answer Phase 1 or the Task 10 DPP protocol) does require HPC and is supported by the bundled SLURM templates.

## Quick start

```bash
make setup              # install Python dependencies
make anonymize-check    # verify the package has no identity leaks
make tests              # run consistency tests (must pass before trusting numbers)
make reproduce-all      # full reproduction: stats + figures + Task 10 DPP + Task 11 audit
```

Outputs land under `results/reproduced/` to keep the bundled `results/` authoritative for comparison. Total wall-clock for `make reproduce-all`: about 30 minutes on a single CPU, longer if `BOOT=10000` is unset.

For a guided walkthrough of every paper-bound number and how to reproduce it, see `REPRODUCIBILITY_GUIDE.md`.

## Repository layout

```
distribution-msv-benchmark/
├── README.md                     (this file)
├── REPRODUCIBILITY_GUIDE.md      (per-claim provenance + step-by-step)
├── LICENSE                       (MIT for code; CC-BY-4.0 for derived metadata)
├── Makefile                      (orchestrates all reproductions)
├── requirements.txt
│
├── data/
│   ├── README.md                 (data provenance, schema, license)
│   ├── gpqa_difficulty_scores.csv
│   ├── kaggle-data/              (23-model Kaggle cohort + Croissant 1.0 metadata)
│   │   ├── kaggle_raw/outputs_logs_corrected.zip
│   │   ├── kaggle_extracted/
│   │   ├── kaggle-msv-benchmark-data.zip
│   │   ├── kaggle-msv-benchmark-data-metadata.json
│   │   └── croissant_metadata.json + croissant_validation.txt
│   ├── forced_answer_phase1/     (institutional 9-model FA outputs, 9 CSVs)
│   ├── task10_dpp/               (institutional 9-model DPP outputs, 9 CSVs + 720 transcripts)
│   └── turing-msv-benchmark-data-metadata.json   (Croissant card for Turing data)
│
├── src/
│   ├── README.md                       (scripts bundle overview)
│   ├── distribution/                   (CPU-only, reviewer-runnable scripts)
│   ├── turing-only/                    (GPU + ollama required)
│   ├── turing-cohort-analyses/         (institutional cohort post-hoc analyses)
│   ├── slurm-templates/                (SLURM job templates)
│   ├── tests/                          (consistency tests; must pass before trusting numbers)
│   ├── internal-utilities/             (defensive QC utilities)
│   ├── utility-scripts/                (audit / diagnostic one-offs)
│   ├── kaggle-notebooks/               (11 verbatim Kaggle competition notebooks)
│   └── task11-notes-writeup-experiments/  (Task 11 audit pipeline + write-up notes)
│
└── results/
    ├── kaggle_cohort/            (Kaggle 23-model analyses, authoritative)
    ├── psychometrics/            (Cronbach's α, bootstrap CI preview)
    ├── task10_dpp/               (Task 10 DPP analysis outputs)
    ├── task11_audit/             (Task 11 audit canonical outputs)
    ├── figures/                  (paper figures: 6 PDFs + 6 PNGs + supporting CSVs)
    └── reproduced/               (`make reproduce-*` outputs land here)
```

## Upstream dataset

The benchmark uses an 80-question subset of **GPQA Diamond** (Rein et al., 2023). Per the GPQA license, the question text is **not redistributed here**. We redistribute:

- The list of 80 question IDs used (column `question_id` in `data/kaggle-data/kaggle_extracted/per_task/t01_delegate_game.csv`).
- Empirical per-question difficulty labels for all 198 GPQA Diamond questions (`data/gpqa_difficulty_scores.csv`), derived from a 10-model open-weight calibration cohort.
- All model outputs on the 80 curated items: 23-model Kaggle cohort outputs in `data/kaggle-data/`, 9-model institutional cohort outputs in `data/forced_answer_phase1/` and `data/task10_dpp/`.

To reproduce runs from upstream raw data, obtain GPQA Diamond from the [original repository](https://github.com/idavidrein/gpqa) and filter to the 80 question IDs in our Task 1 CSV. Section 3.2 of the paper documents the selection procedure.

## What is reproducible from the bundled data

**Fully reproducible end-to-end without HPC** (run `make reproduce-all`):

- All per-task and per-model extracted CSVs for the 23-model Kaggle cohort
- Section 5.2 comparative analysis (own-error and hardness Delegation AUC, rank-reversal numbers, cross-cohort τ = +0.4545 with 95% CI)
- Section 5.4 sensitivity analyses
- Section 5.5 Cronbach's α = 0.978 on Task 4
- Section 5.5 cross-task dissociations and MSV routing subset analysis
- All 6 paper figures: rank-reversal scatter (Section 5.2), risk-coverage curves (Section 5.2), Type-2 AUC vs $\hat{d}$ scatter (Appendix `app:task11_audit`), cross-task convergence heatmap (Appendix `app:task11_audit`), completion heatmap (Appendix), MSV routing subset analysis (Appendix)
- Appendix `app:rank_divergence_audit` (hardness-target sensitivity grid)
- Appendix `app:task11_audit` (Haiku correction to Type-2 AUC = 0.747; five failure modes)
- Appendix `app:task10_dpp_institutional` (matched-paired lift +0.021 with all 9 paired CIs containing zero) — once `compute_task10_dpp_analysis.py` is added

**Reproducible only with local ollama and the open-weight model tags pulled** (see `REPRODUCIBILITY_GUIDE.md` Section 6):

- Re-running the institutional Forced-Answer Phase 1 inference (driver: `src/turing-only/run_forced_answer_phase1_turing.py`)
- Re-running the institutional Task 10 DPP inference at extended context (driver: `src/turing-only/run_task10_dpp_turing.py`)

The pre-computed outputs of those inference runs are bundled, so a reviewer who only wants to verify the *analysis* can do so on a single CPU.

## Pipeline invariants

Four numerical conventions are encoded in code and verified by `src/tests/test_*.py`:

1. **Confidence-to-probability mapping:** `{1, 2, 3, 4} → {0.25, 0.50, 0.75, 1.00}` (paper Section 4.3)
2. **ECE binning:** label-grouped (one bin per discrete confidence level), not equal-width over [0,1] (paper Section 4.3)
3. **Hardness AUC:** uses the global panel median, not per-model medians, for paper-bound results (paper Appendix `app:rank_divergence_audit`)
4. **Bootstrap CIs:** 10,000 question-panel resamples for paper-bound numbers; default `--n_boot 10000`

Run `make tests` first. If either consistency test fails, the local script copy has drifted from the canonical implementation and downstream numbers should not be trusted.

## Dependencies

- Python 3.10+
- numpy ≥ 1.26, pandas ≥ 2.1, scipy ≥ 1.11, scikit-learn ≥ 1.3
- matplotlib ≥ 3.7, seaborn ≥ 0.12 (for the cross-task convergence heatmap)
- requests ≥ 2.31 (only for the Turing driver scripts; not needed for offline analysis)

`requirements.txt` pins tested versions. No GPU code runs in the offline analysis; the Turing driver scripts require a running ollama server with the relevant model tags pulled.

## License

- Code in this repository is released under the **MIT License** (see `LICENSE`).
- The curated 80-question ID list and empirical difficulty labels are released under **CC-BY-4.0** as derivative metadata over GPQA Diamond.
- Model output CSVs (Kaggle cohort and Turing cohort) are released under **MIT** as our own data.
- The GPQA Diamond question text remains under its original license and must be obtained separately from the upstream repository.

## Anonymity notice

This package is prepared for double-blind review. Paths, logs, and metadata have been anonymized. Run `make anonymize-check` to verify; this scans for identity strings, leaky Kaggle paths, OS metadata, and internal provenance files.

Any remaining identifying information in raw Kaggle run archives reflects upstream platform metadata we cannot rewrite without breaking the reconciliation against the Kaggle public leaderboard.

## Citation

Citation block will be added on acceptance.
