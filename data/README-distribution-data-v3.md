# MSV Benchmark — Data README

All datasets used in the paper, organized by source cohort. The Kaggle 23-model cohort and the institutional Turing 9-model cohort are kept separate.

## Contents

```
data/
├── README.md                              (this file)
├── gpqa_difficulty_scores.csv             (198 questions × empirical difficulty)
├── kaggle-data/                           (Kaggle 23-model cohort + Croissant 1.0 metadata)
│   ├── kaggle_raw/
│   │   └── outputs_logs_corrected.zip     (16 MB; 253 nested .zip + .log pairs)
│   ├── kaggle_extracted/
│   │   ├── per_task/                      (11 long-form CSVs)
│   │   ├── per_model/                     (23 model dirs × 11 task CSVs)
│   │   ├── run_metadata.csv               (253 rows: one per model-task run)
│   │   ├── leaderboard_reconciled.csv
│   │   └── extraction_log.txt
│   ├── kaggle-msv-benchmark-data.zip      (Croissant data archive)
│   ├── kaggle-msv-benchmark-data-metadata.json
│   ├── croissant_metadata.json            (Croissant 1.0 with Responsible AI fields)
│   └── croissant_validation.txt
├── forced_answer_phase1/                  (institutional 9-model FA outputs)
│   └── *.csv                              (9 files, 80 rows each)
├── task10_dpp/                            (institutional 9-model DPP outputs)
│   ├── *.csv                              (9 files, 80 rows each)
│   └── *__transcripts/                    (9 dirs × 80 transcript JSONs = 720 total)
└── turing-msv-benchmark-data-metadata.json   (Croissant card for the Turing data)
```

---

## `gpqa_difficulty_scores.csv`

Empirical per-question difficulty for all 198 GPQA Diamond questions. The `difficulty` column is the fraction of a 10-model open-weight calibration cohort that answered the question incorrectly on the full 198-question set.

The 10 calibration models were: `qwen2.5:7b`, `deepseek-r1:7b`, `llama3.1:8b`, `llama3.2:3b`, `phi4:14b`, `gemma2:9b`, `mistral:7b`, `mixtral:8x7b`, `command-r:35b`, `qwen2.5:32b`.

These labels are a derivative work over GPQA Diamond and are released under **CC-BY-4.0**. See Section 3.2 of the paper for the calibration procedure.

The 80-question paper subset is the intersection of this CSV's `question_id` column with the question_id values appearing in `kaggle-data/kaggle_extracted/per_task/t01_delegate_game.csv`.

---

## `kaggle-data/`

The 23-model Kaggle Benchmarks cohort: raw archive, extracted CSVs, and Croissant 1.0 metadata.

### `kaggle_raw/outputs_logs_corrected.zip`

The canonical raw archive of all 253 Kaggle Benchmarks runs (23 models × 11 tasks). Each run contributes one `.zip` and one `.log`:

- The `.zip` contains the task's per-trial results CSV, the task definition JSON, and the full run state JSON from the Kaggle Benchmarks SDK.
- The `.log` is the Kaggle notebook stdout, including the summary line (`Mean score: X | Delegate rate: Y% | Parse failures: Z/N`).

Filenames in this archive use the canonical form `{task}-{task_slug}-{provider}_{model}.{zip|log}` with 38 filename issues from the original upload resolved in place (typos, truncations, separator normalization). See `REPRODUCIBILITY_GUIDE.md` for the rename table.

### `kaggle_extracted/`

Output of `src/distribution/extract_kaggle_outputs.py`. Analysis-ready CSVs for the Kaggle cohort.

#### `per_task/`

One long-form CSV per task, with rows pooled across all 23 models:

| File                        | Rows  | Columns |
|-----------------------------|-------|---------|
| t01_delegate_game.csv       | 1,416 | model, question_id, choice, answer, confidence, correct, difficulty, score, raw_response |
| t02_declared_probe.csv      | 1,460 | model, question_id, category, declared_CE, declared_ER, declared_CI, declared_EM, declared_PI, answer, confidence, correct, choice, score, raw_response |
| t03_second_chance.csv       | 1,360 | model, question_id, phase1_answer, phase1_conf, phase2_answer, phase2_conf, phase2_action, correct, score, raw_response |
| t04_confidence_entropy.csv  | 1,527 | model, question_id, norm_entropy, difficulty, calibration_error, answer_correct, score, raw_response |
| t05_teammate_delegate.csv   | 1,597 | model, question_id, category, teammate_accuracy, choice, answer, confidence, correct, score, raw_response |
| t06_behavioral_er.csv       | 1,255 | model, question_id, framing, answer, confidence, correct, flip, score, raw_response |
| t07_behavioral_ci.csv       | 1,609 | model, question_id, contradiction_type, answer, confidence, correct, score, raw_response |
| t08_behavioral_em.csv       |   639 | model, question_id, wording, answer, confidence, correct, score, raw_response |
| t09_behavioral_pi.csv       | 1,181 | model, question_id, stakes_framing, answer, confidence, choice, correct, score, raw_response |
| t10_dpp_sequence.csv        |   497 | model, question_id, final_letter, correct_letter, correct, stage_failures, context_overflow, *_tokens (5 stage columns), elapsed_s |
| t11_mc_binary_pairs.csv     | 2,735 | model, question_id, is_signal, candidate, judgment, confidence, correct, score, raw_response |

#### `per_model/<model_name>/`

Same data pivoted for per-model consumption. 23 directories, each containing 11 CSVs (one per task). Use this form for per-model analyses.

#### `run_metadata.csv`

One row per (model, task) pair with columns:

- `model`, `task_id`, `task_version`, `run_id`, `run_state`
- `run_result_value` — platform-authoritative mean score (all scheduled trials; missing counted as 0). Matches the Kaggle public leaderboard.
- `log_mean_score` — mean over completed trials only.
- `completed_trials`, `scheduled_trials`, `parse_failures`
- `budget_failure` (bool), `other_failure` (string)

#### `leaderboard_reconciled.csv`

Cross-check between the extracted `run_result_value` and the public Kaggle leaderboard for each (model, task). All 253 rows reconcile within |diff| ≤ 1e-3.

#### `extraction_log.txt`

Log from the extractor run, including alias-table applications and parser warnings.

### Croissant 1.0 metadata

`croissant_metadata.json` provides Croissant 1.0 metadata with Responsible AI fields for the Kaggle data. Validate via:

```bash
make validate-croissant
```

`croissant_validation.txt` records the most recent successful validation. The companion `kaggle-msv-benchmark-data.zip` and `kaggle-msv-benchmark-data-metadata.json` are the Kaggle Datasets-uploadable archive paired with its data card.

---

## `forced_answer_phase1/`

Per-trial outputs of the Forced-Answer Phase 1 inference protocol on the institutional 9-model open-weight cohort. One CSV per model:

- `gemma2_2b.csv`, `gemma2_9b.csv`
- `llama3.1_8b.csv`, `llama3.2_1b.csv`, `llama3.2_3b.csv`
- `mistral_7b.csv`, `phi4-mini_latest.csv`
- `qwen2.5_3b.csv`, `qwen2.5_7b.csv`

Each CSV has 80 rows (one per question) with columns: `question_id, answer, confidence, correct, raw_response, parse_failure`. Used for paper Table 3b (forced-answer same-item comparison) and as the FA baseline in the Task 10 DPP matched-paired analysis (Appendix `app:task10_dpp_institutional`).

The 9-model cohort uses ollama tags. Per-model parse-failure rates we observed: `gemma2:2b` 0%, most other models 2–9%, `llama3.2:1b` 20%, `phi4-mini:latest` 40% (the worst single-shot structured-output compliance in the cohort; see Appendix `app:task10_dpp_institutional` for context).

---

## `task10_dpp/`

Per-trial outputs of the 5-stage Dialectical Deliberative Prompt (DPP) protocol on the same institutional 9-model cohort, run at `NUM_CTX=32768`. Two artifact types per model:

- `<model>.csv` — 80 rows per model with columns: `question_id, final_letter, correct_letter, correct, stage_failures, context_overflow, expert_tokens, critic_tokens, evaluator_tokens, synthesizer_tokens, generalist_tokens, total_prompt_tokens_stage5, elapsed_s`.
- `<model>__transcripts/<question_id>.json` — full per-stage transcripts (Expert / Critic / Evaluator / Synthesizer / Generalist). 80 JSON files per model, 720 total.

All 9 models complete 80/80 questions cleanly: zero context overflows, zero stage failures. Cohort accuracy mean 0.386, range [0.275, 0.488]. Used for paper Appendix `app:task10_dpp_institutional`.

The transcript JSON schema is:
```json
{
  "question": "...",
  "question_id": "...",
  "final_letter": "C",
  "correct_letter": "C",
  "correct": true,
  "stage_failures": 0,
  "context_overflow": false,
  "transcript": {
    "expert": "...",
    "critic": "...",
    "evaluator": "...",
    "synthesizer": "...",
    "generalist": "..."
  }
}
```

---

## `turing-msv-benchmark-data-metadata.json`

Croissant 1.0 metadata card for the institutional Turing data (`forced_answer_phase1/` + `task10_dpp/`). Documents:

- The 9 models with ollama tag versions
- The 80-question subset (pointer to `gpqa_difficulty_scores.csv`)
- Per-trial schema for FA Phase 1 CSVs and Task 10 DPP CSVs/transcripts
- License: MIT for the model outputs (we hold copyright on what our scripts produced)
- Date of collection: 2026-04-25 (FA Phase 1) and 2026-04-26 (Task 10 DPP)

Validate via `make validate-croissant` (validates both Kaggle and Turing cards).

---

## Scoring convention (important)

Two different "mean scores" appear in the extracted Kaggle data:

- `log_mean_score` — mean over completed trials only.
- `run_result_value` — mean over all scheduled trials (missing = 0). This is the platform-authoritative Kaggle leaderboard number.

For any per-trial analysis (rank reversals, sensitivity, item discrimination) use only completed trials. For overall leaderboard comparisons use `run_result_value`.

`deepseek-r1-0528` on Task 1 exemplifies the divergence: 2 of 80 trials completed, both scored 1.0. `log_mean_score = 1.0` but `run_result_value = 0.025`. Use the appropriate column for the analysis you are running.

---

## Completion picture

Of the 253 task-model Kaggle runs:

- 133 completed cleanly
- 50 hit budget or quota limits partway through
- 70 hit other platform errors (API 503, 400, timeouts, context-length)

See `results/kaggle_cohort/completion_adjusted_summary.csv` for per-model completion rates alongside raw vs platform-adjusted mean scores. The Task 10 column is near-universally red because the Kaggle SDK's default `num_ctx` is too small for 5-stage DPP prompt accumulation; the institutional cohort at `NUM_CTX=32768` has 9/9 clean completions, addressing the platform limitation.

---

## Upstream question text (NOT included)

The GPQA Diamond question text is **not redistributed here**. To obtain it, clone https://github.com/idavidrein/gpqa and filter to the 80 question IDs in our Task 1 CSV's `question_id` column.

The 80-question subset is also identified by the union of:
- `kaggle-data/kaggle_extracted/per_task/t01_delegate_game.csv` (column `question_id`)
- The transcript filenames in `task10_dpp/*__transcripts/`

Both sources should agree on the 80-question membership.

---

## License

- Empirical difficulty labels (`gpqa_difficulty_scores.csv`) and curated 80-question ID list: **CC-BY-4.0** (derivative metadata).
- Per-trial model outputs (Kaggle CSVs, FA Phase 1 CSVs, Task 10 DPP CSVs and transcripts): **MIT** (our own data).
- GPQA Diamond questions: upstream license; not redistributed.
