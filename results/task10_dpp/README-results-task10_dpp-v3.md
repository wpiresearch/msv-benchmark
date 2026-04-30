# Task 10 DPP Analysis Outputs

This directory holds the canonical CSVs for paper Appendix `app:task10_dpp_institutional` and Limitation 11 (matched-lift CIs and the per-model MDE that defends the negative result). Files are produced by two scripts in `src/distribution/`:

- `compute_task10_dpp_analysis.py` produces `task10_lift.csv`, `task10_winloss.csv`, `task10_trace.csv`, `task10_extract_qc.csv`
- `compute_task10_mde.py` produces `task10_mde.csv`

Both scripts read from:

- `data/task10_dpp/` — 9 per-model DPP CSVs + 720 transcript JSONs
- `data/forced_answer_phase1/` — 9 per-model Forced-Answer Phase 1 CSVs
- `data/gpqa_difficulty_scores.csv` — empirical difficulty labels (consistency probe; not used for stratification)

To regenerate from scratch:

```bash
make reproduce-task10-dpp
```

Wall-clock: ~10 seconds with `BOOT=10000`.

## Files

### `task10_lift.csv`

Per-model end-to-end and matched-paired lift with paired-bootstrap 95% CIs.

| Column | Meaning |
|---|---|
| `model` | Model name (matches DPP and FA CSV stems). Special values `COHORT_MEAN_OF_MODELS` (paper convention; unweighted mean of per-model lifts) and `COHORT_POOLED` (n-weighted pooled-diff bootstrap) for cohort-aggregate rows |
| `n_dpp` | DPP CSV row count (always 80; no stage failures, no overflows in this cohort) |
| `n_fa_total` | FA Phase 1 CSV row count (always 80) |
| `n_fa_parsed` | FA rows where `parse_failure == 0` |
| `fa_parse_rate` | `n_fa_parsed / n_fa_total` |
| `n_matched` | Inner join of DPP and FA-parsed by `question_id` |
| `dpp_acc` | Mean DPP correctness (full panel) |
| `fa_acc_full` | Mean FA correctness (full panel; parse-failures count as wrong) |
| `fa_acc_parsed` | Mean FA correctness (parsed subset only) |
| `end_to_end_lift` | `dpp_acc - fa_acc_full`, no matching |
| `matched_lift` | `mean(DPP_correct - FA_correct)` over the matched subset |
| `matched_lift_ci_low` / `matched_lift_ci_high` | Paired-bootstrap 95% percentile CI (B = `n_boot`, seed=42) |
| `n_boot` | Bootstrap iteration count used |
| `ci_excludes_zero` | 1 if CI is entirely above or below zero, 0 otherwise. **All 9 are 0** in this cohort; that's the headline negative result |

Two cohort-aggregate rows:
- `COHORT_MEAN_OF_MODELS`: unweighted mean of per-model lifts. **+0.021, 95% CI [-0.027, +0.071].** Each model contributes equally; this is what the paper Appendix table reports.
- `COHORT_POOLED`: pooled-diff bootstrap on the concatenated question-level diff vector. +0.017, 95% CI [-0.029, +0.063]. Larger-n models dominate.

Both are reported so the convention is explicit.

### `task10_winloss.csv`

Per-model paired wins/losses on the matched subset, plus McNemar exact two-sided p.

| Column | Meaning |
|---|---|
| `model` | Model name; `COHORT_AGGREGATE` for the cohort-sum row |
| `n_matched` | Same as in `task10_lift.csv` |
| `wins` | Count of (DPP correct, FA wrong) |
| `losses` | Count of (DPP wrong, FA correct) |
| `tie_correct` | Count of (both correct) |
| `tie_wrong` | Count of (both wrong) |
| `discordant` | `wins + losses` |
| `churn_rate` | `discordant / n_matched` |
| `mcnemar_exact_p` | Two-sided exact McNemar via `scipy.stats.binomtest(min(wins,losses), wins+losses, p=0.5)` |
| `mcnemar_significant_05` | 1 if `p < 0.05`, 0 otherwise. **All 9 are 0**; only llama3.1:8b and gemma2:2b have p < 0.15 |

Cohort aggregate row sums across models. Paper-locked headline: 120 wins, 109 losses, 229 discordant, 648 matched, churn 0.354.

### `task10_trace.csv`

Per-model Expert→Generalist trace correction analysis.

| Column | Meaning |
|---|---|
| `model` | Model name |
| `n_total_transcripts` | Number of transcripts read (always 80 per model) |
| `n_expert_extracted` | Number where the layered regex extracted an Expert-stage final letter |
| `expert_extract_rate` | `n_expert_extracted / n_total_transcripts` |
| `low_extract` | 1 if `expert_extract_rate < 0.80`, else 0. **Flagged models** (paper convention: prose excludes them, table retains with †): llama3.2:1b (0.45), mistral:7b (0.69), qwen2.5:3b (0.61) |
| `expert_acc` | Mean Expert-stage correctness over the extracted subset |
| `final_acc_on_extracted` | Mean Generalist-stage (= final_letter from CSV) correctness on the same subset |
| `rescues` | Count of (Expert wrong, Final correct) — DPP corrected the Expert |
| `harms` | Count of (Expert correct, Final wrong) — DPP introduced an error |
| `net_correction` | `rescues - harms` |
| `stable_correct` | Count of (both correct) |
| `stable_wrong` | Count of (both wrong) |
| `binomial_two_sided_p` | Two-sided binomial test on (rescues vs harms) under H0: p=0.5. Matches paper's "≈p" notation |

Paper-locked highlights:
- **llama3.1:8b**: 16 rescues, 7 harms, net +9, p ≈ 0.093 (closest to significance)
- **qwen2.5:7b**: 6 rescues, 8 harms, net -2, p = 0.79
- Models with `low_extract=1` should be reported with † footnote in any table

### `task10_extract_qc.csv`

Quality-control summary for the Expert-letter regex extractor.

| Column | Meaning |
|---|---|
| `model` | Model name |
| `n_total_transcripts` | 80 |
| `n_expert_extracted` | Same as in `task10_trace.csv` |
| `expert_extract_rate` | Same as in `task10_trace.csv` |
| `low_extract_flag` | Same as `low_extract` in `task10_trace.csv` |
| `top_pattern_hits` | Top 5 layered-regex patterns by hit count, semicolon-separated. Includes `unmatched=N` for the count of items where no pattern matched (these are mostly free-form prose responses that never explicitly state a letter) |

Use this CSV to debug regex coverage if extending the analysis to a new cohort. The paper does not cite this CSV directly but it provides the audit trail for why certain models carry the † flag.

### `task10_mde.csv`

Per-model **minimum detectable effect (MDE)** for the matched-lift analysis. Quantifies what magnitude of per-model lift the 80-item panel could have detected at conventional significance levels, given each model's empirical paired-diff variability. Cited in paper Limitation 1, Limitation 11, and Appendix `app:extended_results`.

| Column | Meaning |
|---|---|
| `model` | Model name. Special row `COHORT_POOLED` for the pooled-diff aggregate |
| `n_matched` | Inner join of DPP and FA-parsed (matches `n_matched` in `task10_lift.csv`) |
| `observed_lift` | Mean of the paired-diff vector `DPP_correct - FA_correct`. Matches `matched_lift` in `task10_lift.csv` to the rounding precision |
| `observed_sd` | Sample SD of the paired-diff vector (ddof=1) |
| `observed_se` | `observed_sd / sqrt(n_matched)` |
| `mde_two_sided_05` | `1.959964 * observed_se`. Two-sided detection threshold at α = 0.05 — the magnitude the panel could have detected with 95% confidence |
| `mde_two_sided_10` | `1.644854 * observed_se`. Two-sided detection threshold at α = 0.10 |
| `mde_one_sided_05` | `1.644854 * observed_se`. One-sided detection threshold at α = 0.05 |
| `observed_below_mde_05_twosided` | 1 if `abs(observed_lift) < mde_two_sided_05`, else 0. **8 of 9 models are 1**; gemma2:2b is the only 0 (observed +0.113 vs MDE 0.110 — at threshold) |

Paper-locked headline values:
- **Per-model MDE range: 0.110 to 0.190** (depends on paired-diff SD and n_matched)
- **Cohort-pooled MDE: 0.046** — the panel could have detected an aggregate lift of about 5 percentage points
- **Only `gemma2:2b` sits at the detectability threshold** (consistent with its near-significant McNemar p = 0.078)

Interpretation: the 80-item panel had power to detect lifts of ~13 percentage points per model and ~5 percentage points pooled, but observed effects fell well below those thresholds. The MDE complements the bootstrap-CI result by establishing that the negative result is panel-power-bounded rather than evidence of a non-zero effect the panel was too small to find.

## Headline cohort results (locked)

The values in these CSVs reproduce paper Appendix `app:task10_dpp_institutional` table values. Cohort-aggregate values (paper convention):

- **Cohort mean lift (unweighted): +0.021, 95% CI [-0.027, +0.071]** (over per-model bootstrap)
- **Lift range across models: [-0.078, +0.125]** (mistral:7b lowest, gemma2:2b highest)
- **0 of 9 paired-bootstrap 95% CIs exclude zero** — this is the headline negative result for the protocol's per-model effect
- **Cohort win/loss: 120 wins / 109 losses / 229 discordant / 648 matched / 0.354 churn**
- **No model reaches McNemar exact p < 0.05** (closest: gemma2:2b at 0.078, llama3.1:8b at 0.108)
- **All 9 models complete 80/80 questions cleanly** — zero context overflows, zero stage failures (the operational feasibility finding for `NUM_CTX=32768`)
- **Cohort-pooled MDE = 0.046; 8 of 9 models had observed |lift| < per-model MDE** — the panel-power-bounded null

Trace-correction headline:
- llama3.1:8b shows the **largest positive net correction** (+9 = 16 rescues − 7 harms; binomial p ≈ 0.093)
- qwen2.5:3b shows the **clearest negative balance** (net −3, but flagged with `low_extract=1`)
- For unflagged models (extract_rate ≥ 0.80), the trace numbers can be cited directly; flagged models should be noted with †.

## Verification: script outputs match paper-locked values

The two scripts reproduce every paper-bound number in Appendix `app:task10_dpp_institutional` and Limitations 1 and 11:

| Source CSV | Paper-locked value | Script output | Match |
|---|---|---|---|
| `task10_lift.csv` | Cohort mean lift +0.021 | +0.0213 | ✓ exact |
| `task10_lift.csv` | Lift range [-0.078, +0.125] | [-0.0779, +0.1250] | ✓ exact |
| `task10_lift.csv` | 0 of 9 paired-bootstrap CIs exclude zero | `ci_excludes_zero=0` for all 9 | ✓ exact |
| `task10_winloss.csv` | 120 wins, 109 losses, 229 discordant, 648 matched, 0.354 churn | 120, 109, 229, 648, 0.3534 | ✓ exact |
| `task10_winloss.csv` | llama3.1:8b McNemar exact p = 0.108 | 0.1078 | ✓ exact |
| `task10_trace.csv` | llama3.1:8b net +9 (16 rescues, 7 harms), binomial p ≈ 0.093 | +9, 0.0931 | ✓ exact |
| `task10_mde.csv` | Cohort-pooled MDE = 0.046 | 0.0458 | ✓ exact |
| `task10_mde.csv` | gemma2:2b observed +0.113 vs MDE 0.110 (at threshold) | obs +0.1125, MDE 0.1102 | ✓ exact |
| `task10_mde.csv` | 8 of 9 models have \|observed lift\| < per-model MDE | `observed_below_mde_05_twosided=1` for 8/9 | ✓ exact |

Bit-for-bit match is expected up to the rounding precision shown in the paper. If a number doesn't match, first verify that `make tests` passes (the four pipeline invariants documented in `REPRODUCIBILITY_GUIDE.md` Section 4); then check `n_boot` and `seed` (defaults: 10000 and 42).

## Implementation notes

A few non-obvious convention choices that reviewers may want to know:

- **Two-sided binomial throughout.** Both the McNemar exact test (`task10_winloss.csv`) and the rescue-vs-harm test (`task10_trace.csv`) use `scipy.stats.binomtest(...).pvalue` with the default two-sided alternative. This matches the paper's "≈p" notation (e.g., llama3.1:8b's binomial p ≈ 0.093 is the two-sided value; the one-sided value would be ≈ 0.047 and would mislead readers into seeing significance that isn't there at α = 0.05 two-sided).
- **Layered regex for Expert-stage extraction.** The 9 institutional models produce diverse free-form Expert-stage outputs ("Answer: X", "X)", "**A/B/C/D**: X", "The correct answer is X", bare letter at line start, "(X)" anywhere in the first 200 characters). `compute_task10_dpp_analysis.py` applies 11 patterns in priority order; the last is a permissive fallback. Per-model hit counts are written to `task10_extract_qc.csv` so reviewers can audit coverage.
- **Models with extraction rate < 0.80 are flagged but not silenced.** Per `task10_extract_qc.csv`, llama3.2:1b (0.45), mistral:7b (0.69), and qwen2.5:3b (0.61) fall below threshold. The paper convention is to display them in tables with † footnotes (rather than dropping them); the `low_extract=1` flag in `task10_trace.csv` marks the rows.
- **MDE uses standard normal critical values.** `mde_two_sided_05 = 1.959964 * SE` (`z_{0.975}`); `mde_two_sided_10 = 1.644854 * SE` (`z_{0.95}`); `mde_one_sided_05` is identical to two-sided 0.10 by symmetry. Per-model SE comes from the FA-parsed inner-join sample (no pooled-variance adjustment); the cohort-pooled row concatenates per-model paired-diff vectors with equal per-trial weight (not meta-analytic SE^-2 weighting).

## Cross-references

- Source data: `data/task10_dpp/` (CSVs and transcripts) + `data/forced_answer_phase1/` (FA Phase 1)
- Generation scripts: `src/distribution/compute_task10_dpp_analysis.py` and `src/distribution/compute_task10_mde.py`
- Reproduction guide: `REPRODUCIBILITY_GUIDE.md` Section 5 (step-by-step), Section 9 (psychometric / detectability analyses including MDE)
- Paper sections: Appendix `app:task10_dpp_institutional` (lift, win/loss, trace); Limitation 1 / Limitation 11 / Appendix `app:extended_results` (MDE)
