# Psychometric Reliability Outputs

This directory holds psychometric reliability statistics on the MSV Benchmark task suite. All files are bit-for-bit reproducible from `data/kaggle-data/kaggle_extracted/per_task/` via scripts in `src/distribution/`.

The paper cites these files in:

- Section 5.5 — Cronbach α reliability claim for Task 4 (paragraph on companion-task reliability)
- Limitation 8 — empirical evidence for partial construct validity of Tasks 6–9 (Task 7 high α, Task 8 negative α)
- Limitation 10 — per-task α reliability tiers across the 11-task suite
- Appendix `app:extended_results` — full per-task α table

## Files

### `cronbach_alpha_task4.csv` and `cronbach_alpha_task4.txt`

Single-task Cronbach α for Task 4 (Confidence Entropy), kept for backward compatibility with earlier paper drafts. Produced by `src/distribution/compute_cronbach_alpha_task4.py`.

| File | Format | Contents |
|---|---|---|
| `cronbach_alpha_task4.csv` | CSV | Pooled and per-domain α at thresholds t = 23, 20, 15. Columns: `domain`, `threshold`, `n_items`, `n_models`, `alpha` |
| `cronbach_alpha_task4.txt` | Plain text | Human-readable summary |

Headline: **pooled strict α = 0.978** (k = 23 items, n = 23 models). Per-domain α at threshold t = 15 spans 0.310 (genetics) to 0.925 (organic chemistry).

### `cronbach_alpha_all_tasks.csv` and `cronbach_alpha_all_tasks.txt`

Full per-task Cronbach α across the 11-task suite, computed under the same convention as the Task 4 script (items × respondents matrix at three balance conventions, optional per-domain stratification). Produced by `src/distribution/compute_cronbach_alpha_all_tasks.py`. Reproduces Task 4 strict α = 0.978 exactly as a validation check.

| File | Format | Contents |
|---|---|---|
| `cronbach_alpha_all_tasks.csv` | CSV | Wide table: task × threshold × domain → α. Columns: `task`, `task_stem`, `domain`, `threshold`, `n_items`, `n_models`, `alpha` |
| `cronbach_alpha_all_tasks.txt` | Plain text | Human-readable summary, pooled α by task and per-domain α at threshold t = 15 |

**Headline values** (pooled at three balance conventions):

| Task | α (t = 23 strict) | α (t = 15 inclusive) | Reliability tier |
|---|---|---|---|
| Task 1 (Delegate Game)         | 0.575 | 0.875 | acceptable |
| Task 2 (Declared MSV Probe)    | 0.974 | 0.989 | excellent |
| Task 3 (Second-Chance)         | 0.666 | 0.879 | acceptable |
| Task 4 (Confidence Entropy)    | 0.978 | 0.976 | excellent |
| Task 5 (Teammate Delegate)     | 0.755 | 0.924 | good |
| Task 6 (Behavioral ER)         | 0.723 | 0.958 | good |
| Task 7 (Behavioral CI)         | 0.965 | 0.994 | excellent |
| **Task 8 (Behavioral EM)**     | **−0.000** | **−1.571** | **construct-degenerate** |
| Task 9 (Behavioral PI)         | n/a (k=1) | 0.995 | excellent (when defined) |
| Task 10 (DPP Sequence)         | 0.365 | 0.018 | uninterpretable (low k) |
| Task 11 (MC Binary Pairs)      | 0.530 | 0.869 | moderate to good |

The Task 8 negative α at every threshold is a substantive finding rather than a power issue: items measuring the EM construct produce inversely correlated responses across models, consistent with the proxy-validity caveat reported in Section 3.3 (Tasks 6–10 paragraph) and strengthening Limitation 8.

Per-task notes:

- **Task 2** uses the `routing_score` column (the principal Task 2 metric combining parseability, differentiation, and routing alignment)
- **Task 8** is indexed by pair `id` rather than `question_id`; per-domain stratification is therefore not available
- **Tasks 9, 10** have insufficient strict-panel completion to produce interpretable α at t = 23
- **Task 11** uses `judgment_correct` aggregated to per-question accuracy (mean of the 2 trials per question, signal + lure)

### `bootstrap_ci_summary_n500_kaggle_preview.csv`

Pre-published preview bootstrap CI summary on the Kaggle cohort (n_boot = 500). Used during paper development for fast iteration; the paper-bound numbers use n_boot = 10,000 from `compute_bootstrap_ci.py` (output not in this directory; see `results/kaggle_cohort/`). This file is preserved for provenance only.

## Three balance conventions

Cronbach α is defined on a complete items × respondents matrix. The Kaggle cohort has 23 models × variable per-task completion; we adopt three balance conventions to handle missing cells:

1. **t = 23 (strict balanced panel):** keep only items completed by all 23 models. Most conservative; matches textbook α definitions.
2. **t = 20 (relaxed panel):** keep items completed by ≥ 20 of 23 models, then drop any models with NaN on the surviving items. Larger k.
3. **t = 15 (inclusive panel):** keep items completed by ≥ 15 of 23 models. Largest k; most generous.

The strict-panel α is the canonical value; the relaxed and inclusive thresholds let reviewers see how the point estimate changes with completion sensitivity.

## Per-domain stratification

Tasks indexed by GPQA `question_id` get per-domain α reported alongside pooled α. Categories are joined from `t02_declared_probe.csv` (10 GPQA Diamond subdomains: astrophysics, chemistry_general, electromagnetism_and_photonics, genetics, high-energy_particle_physics, molecular_biology, organic_chemistry, physics_general, quantum_mechanics, relativistic_mechanics).

## Regenerating

```bash
make reproduce-stats
```

This invokes both `compute_cronbach_alpha_task4.py` and `compute_cronbach_alpha_all_tasks.py`. Outputs land in `results/reproduced/`. Compare bit-for-bit against the bundled `results/psychometrics/`.

## Cross-references

- Source data: `data/kaggle-data/kaggle_extracted/per_task/`
- Generation scripts: `src/distribution/compute_cronbach_alpha_task4.py` and `src/distribution/compute_cronbach_alpha_all_tasks.py`
- Reproduction guide: `REPRODUCIBILITY_GUIDE.md` Section 9 (psychometric reliability and detectability analyses)
- Paper sections: Section 5.5; Limitations 8 and 10; Appendix `app:extended_results`
