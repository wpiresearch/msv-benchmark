# turing-cohort-analyses/

Three complementary analysis scripts that target the **9-model Turing
institutional cohort** specifically. They are NOT subsumed by
`distribution/analyze_kaggle_cohort.py` — that script targets the
**23-model Kaggle cohort** and produces Kaggle-specific output paths.

These scripts are explicitly promised in paper Appendix
`app:reproducibility` (line 1183 of `main-ed-v2_5.tex`):

> "(d) all analysis scripts (comparative baselines, **bootstrap CIs,
> sensitivity analysis, ranking stability, synthetic delegation
> baseline**), ..."

## Scripts and their paper destinations

### compute_sensitivity_analysis.py

**Where it's subsumed (Kaggle cohort):** `distribution/analyze_kaggle_cohort.py`
produces the equivalent for the 23-model Kaggle cohort, output as
`results/.../sensitivity/task1_reward_sensitivity.csv` and
`results/.../sensitivity/task1_difficulty_sensitivity.csv`. These
populate Appendix Tables `tab:kaggle_reward_sens`, `tab:kaggle_diff_sens`,
and `tab:kaggle_per_model_reward` (paper lines 636-704).

**Where it's NOT subsumed (Turing cohort):** Section 5.4 main body
(paper line 327-340) reports sensitivity numbers on the 9-model Turing
cohort separately from the appendix tables. From the paper text:

> "The corresponding analysis on the 9-model Turing cohort is reported
> in the main body (Section~\ref{sec:sensitivity}); summary numbers
> are cited there."

This script produces those Turing-cohort numbers. Standalone analysis
on the 9-model cohort produces:

- `reward_sensitivity_summary.csv` — per-perturbation rank correlations
- `reward_sensitivity_rankings.csv` — model rankings under each schedule
- `difficulty_sensitivity_summary.csv` — rank correlations under threshold changes
- `difficulty_sensitivity_rankings.csv` — model rankings under each threshold

Usage:

```bash
python compute_sensitivity_analysis.py \
    --input_dir   results/reproduced/turing_analysis_input_80q/delegate_game/ \
    --output_dir  results/reproduced/sensitivity_turing/
```

### compute_ranking_stability.py

**Where it's subsumed (Kaggle cohort):** `distribution/analyze_kaggle_cohort.py`
produces the equivalent for the 23-model Kaggle cohort, output as
`results/.../stability/task1_ranking_stability_summary.csv`,
`task1_ranking_stability_distribution.csv`, and
`task1_item_discrimination.csv`.

**Where it's NOT subsumed (Turing cohort):** Same as above — the paper's
"split-half ranking stability" claim (line 81, line 419) on the 9-model
Turing cohort is computed separately from the Kaggle cohort. This
script produces those Turing-cohort numbers.

This script also **supersedes** the older `compute_split_half_reliability.py`
(dropped from this bundle); see the docstring for the rename rationale.

Usage:

```bash
python compute_ranking_stability.py \
    --input_dir   results/reproduced/turing_analysis_input_80q/delegate_game/ \
    --output_dir  results/reproduced/stability_turing/ \
    --n_splits    1000
```

### compute_synthetic_delegation.py

**Not subsumed by any other script.** This is the only script that
implements the synthetic-delegation comparison: confidence-thresholded
"would-have-delegated" labels from forced-answer Phase 1 confidence
ratings, compared against actual Delegate Game Phase 2 delegation
decisions via Cohen's κ.

The v2 design (in this bundle) requires forced-answer Phase 1 data,
which we now have. With forced-answer data, the script computes
synthetic delegation as `delegate_synthetic = (forced_answer_confidence < t)`
for several thresholds t, then computes Cohen's κ between
`delegate_synthetic` and `delegate_actual` per model.

This is a key analysis for the paper's central thesis: if κ is low,
behavioral delegation captures information not present in stated
confidence (one of the headline claims of Section 5.2).

Usage:

```bash
python compute_synthetic_delegation.py \
    --delegate_dir       results/reproduced/turing_analysis_input_80q/delegate_game/ \
    --forced_answer_dir  results/results-gpqa-2026-03-25/forced_answer_phase1/ \
    --output_dir         results/reproduced/synthetic/
```

This script has not yet been run on the v9 forced-answer data; running
it is a recommended next step before final paper integration.

## Why this folder exists

The original v8/v9 bundle proposal lumped these scripts as either
"subsumed" (and proposed dropping) or "extended-analyses" (vague).
Both framings were wrong:

- They are NOT subsumed for the Turing cohort
- They are NOT extended/optional — they are explicitly named in the
  paper's Reproducibility appendix as required artifacts

The accurate framing is "complementary analyses on the Turing cohort,
parallel to what `analyze_kaggle_cohort.py` does for the Kaggle cohort,
required for full reproducibility of paper claims."
