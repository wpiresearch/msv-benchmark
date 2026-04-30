# Testing the Metacognitive Inefficiency Hypothesis: Analysis Plan

This document describes how to further test and characterize the metacognitive inefficiency pattern observed in reasoning-enhanced LLMs (Claude Haiku 4.5, GLM-5) on Task 11, and the related zero-delegation behavior observed on Task 1 (Gemma 4 31B, Gemini 2.5 Flash, DeepSeek R1, GLM-5). The goal is to turn the current two-model observation into a statistically defensible finding that belongs in the NeurIPS paper.

All analyses described here operate on files already saved to disk during Kaggle runs: per-task CSVs in `/kaggle/working/` and per-model JSON run files (`*-run_id_Run_1_<model>.run.json`). No new model calls are required.

---

## 1. What we have and what we need

### What Kaggle runs save per task
- **`tXX_*_results.csv`**: per-question records with `question_id`, `score`, and task-specific fields. Task 1 includes `choice` (ANSWER/DELEGATE), `answer`, `confidence`, `correct`, and `difficulty`. Task 11 includes per-trial hit/FA flags and confidence ratings. Task 2 includes declared MSV values and routing decisions.
- **`tXX-*-run_id_Run_1_<model>.run.json`**: per-request metadata for every prompt in the run, including `inputTokens`, `outputTokens`, `totalBackendLatencyMs`, and the full prompt and response text. This file is the gold mine for post-hoc analysis: everything needed to measure output verbosity, reconstruct the full conversation, and compute per-response statistics is here.

### What we need to build
- A small library of Python analysis scripts that consume these files and produce both summary numbers and figures.
- Each script should be standalone, take a directory of run files as input, and produce either a CSV of summary statistics or a figure saved to disk.

### Directory layout assumption
```
/analysis/
  data/
    kaggle_runs/
      t01-msv_delegate_game-run_id_Run_1_<model>.run.json  (×N models)
      t02-msv_declared_probe-run_id_Run_1_<model>.run.json
      t11-msv_mc_binary_pairs-run_id_Run_1_<model>.run.json
    kaggle_csvs/
      t01_delegate_game_results_<model>.csv  (×N models)
      t02_declared_probe_results_<model>.csv
      t11_mc_binary_pairs_results_<model>.csv
  scripts/
    (Python analysis scripts described below)
  outputs/
    (CSVs and figures produced by scripts)
```

You may need to rename files after download to include the model slug, since Kaggle saves them with generic names. A trivial rename wrapper is part of Script 0 below.

---

## 2. The hypotheses to test

The NeurIPS section advances one primary hypothesis with three supporting observations. The testing plan below addresses each.

**H1 (primary):** Reasoning-enhanced models exhibit metacognitive inefficiency: substantial object-level discrimination paired with chance-level metacognitive discrimination.

**H2 (supporting):** The zero-delegation behavior observed on Task 1 for reasoning models is not a measurement artifact but reflects flat routing across difficulty levels.

**H3 (supporting):** The declared-vs-behavioral mismatch on Task 2 (model produces differentiated MSV self-reports but does not translate them into routing action) is specific to reasoning models.

**H4 (secondary):** The verbose-CoT failure pattern on the Kaggle SDK is correlated with reasoning-training, and the degree of verbosity correlates with the severity of the metacognitive-inefficiency pattern within reasoning models.

Each hypothesis maps to one or more scripts below.

---

## 3. Analysis scripts

### Script 0: File organization and catalog

Produce a catalog CSV listing every available run file with its model, task, completion count, and file path. This becomes the manifest that all other scripts iterate over.

**Inputs:** directory of raw run files from Kaggle.
**Outputs:** `run_catalog.csv` with columns `model`, `task`, `n_requests`, `n_questions_total`, `completion_rate`, `run_file_path`, `csv_file_path`.

**Why this matters:** every subsequent script needs to know which models completed which tasks. Keeping a central manifest avoids drift between scripts.

---

### Script 1: Verbosity analysis from run files (H4)

For each run file, compute distributional statistics over the per-request `outputTokens` field.

**Inputs:** `*.run.json` files.
**Outputs:** `verbosity_stats.csv` with columns `model`, `task`, `n_requests`, `output_tokens_mean`, `output_tokens_median`, `output_tokens_p90`, `output_tokens_max`, `latency_ms_mean`.

Plus a figure: **output-token distribution by model, ordered from terse to verbose**, with a vertical line at the 2K-token threshold separating "followed JSON-only instructions" from "emitted verbose CoT."

**Implementation sketch:**
```python
import json
import pandas as pd
from pathlib import Path
import numpy as np

def summarize_run(run_file):
    with open(run_file) as f:
        data = json.load(f)
    reqs = data['conversations'][0]['requests']
    out_tokens = [r['metrics']['outputTokens'] for r in reqs]
    latencies = [int(r['metrics']['totalBackendLatencyMs']) for r in reqs]
    return {
        'n_requests': len(reqs),
        'output_tokens_mean': np.mean(out_tokens),
        'output_tokens_median': np.median(out_tokens),
        'output_tokens_p90': np.percentile(out_tokens, 90),
        'output_tokens_max': np.max(out_tokens),
        'latency_ms_mean': np.mean(latencies),
    }
```

**What we expect to see:** reasoning models (DeepSeek R1, GLM-5, Gemini 2.5 Flash) produce median output-token counts an order of magnitude larger than non-reasoning models (Claude Haiku, GPT 5.4). If the histogram shows a clean bimodal distribution with no intermediate models, that is publishable on its own as evidence that "reasoning model" is a measurable behavioral category on this benchmark.

---

### Script 2: Type-2 AUC and MC recomputation with confidence intervals (H1)

Task 11's online score is computed on the fly during the run and does not include confidence intervals on $d^{*}$ or MC. Recompute these offline with bootstrap resampling.

**Inputs:** `t11_mc_binary_pairs_results_<model>.csv` files.
**Outputs:** `task11_metacognitive_efficiency.csv` with columns `model`, `n_trials`, `d_hat`, `d_hat_ci_low`, `d_hat_ci_high`, `type2_auc`, `type2_auc_ci_low`, `type2_auc_ci_high`, `mc`, `mc_ci_low`, `mc_ci_high`.

Plus a figure: **scatter plot of $\hat{d}$ (x-axis) vs type-2 AUC (y-axis), one point per model**, with bootstrap confidence ellipses. Reasoning models should cluster in the "high $\hat{d}$, 0.5 type-2 AUC" quadrant. Non-reasoning models should fall along the diagonal (more typical pattern where metacognitive discrimination tracks object-level discrimination).

**Implementation sketch:**
```python
from sklearn.metrics import roc_auc_score
from scipy.stats import norm

def compute_d_hat(hits, signal_trials, false_alarms, noise_trials):
    # Standard SDT with hit/FA rate clipping to avoid infinities
    h = max(0.01, min(0.99, hits / signal_trials))
    f = max(0.01, min(0.99, false_alarms / noise_trials))
    return norm.ppf(h) - norm.ppf(f)

def bootstrap_mc(df, n_boot=1000):
    # Resample trials with replacement; recompute d_hat, type-2 AUC, MC
    mcs = []
    for _ in range(n_boot):
        sample = df.sample(len(df), replace=True)
        # ... compute d_hat, d_star, mc from sample ...
        mcs.append(mc)
    return np.percentile(mcs, [2.5, 50, 97.5])
```

**What we expect to see:** the current two-model pattern holds after bootstrap, with 95% confidence intervals on type-2 AUC for reasoning models that exclude values above ~0.55. If the confidence intervals remain tight at chance level, this upgrades the finding from "suggestive" to "robust on the available sample."

---

### Script 3: Delegation-rate-by-difficulty curve (H2)

Task 1's scoring schedule rewards delegation on hard questions and penalizes it on easy ones. A metacognitively functional model should show a monotonically increasing delegation rate with difficulty. A flat zero-delegation curve across all difficulty levels indicates the routing signal carries no information.

**Inputs:** `t01_delegate_game_results_<model>.csv` files.
**Outputs:** `task1_delegation_curves.csv` with one row per (model, difficulty-bin) pair, columns `model`, `difficulty_bin`, `n_questions`, `delegate_rate`, `mean_score`.

Plus a figure: **delegation rate as a function of difficulty, one line per model**, with reasoning models (expected to be flat at zero) rendered in one color and non-reasoning models (expected to show positive slope) in another.

**Implementation sketch:**
```python
def delegation_by_difficulty(df, n_bins=5):
    df = df.copy()
    df['diff_bin'] = pd.qcut(df['difficulty'], n_bins, labels=False)
    return df.groupby('diff_bin').agg(
        n=('question_id', 'count'),
        delegate_rate=('choice', lambda x: (x == 'DELEGATE').mean()),
        mean_score=('score', 'mean'),
    )
```

**What we expect to see:** non-reasoning models (Claude Haiku, GPT 5.4) show delegation rates that increase from ~10% on easy questions to ~40%+ on the hardest questions. Reasoning models (Gemma 4 31B, Gemini 2.5 Flash, DeepSeek R1, GLM-5) show flat curves near zero across all difficulty bins. The visual contrast makes the H2 claim immediate.

---

### Script 4: Declared-vs-behavioral coherence on Task 2 (H3)

Compare the declared MSV activation values in Task 2 with the routing decision chosen by the model on the same question. A metacognitively coherent model should choose DELEGATE or DELIBERATE when declared activation is high, ANSWER when low.

**Inputs:** `t02_declared_probe_results_<model>.csv` files.
**Outputs:** `task2_coherence.csv` with columns `model`, `activation_mean`, `activation_std`, `delegate_rate`, `deliberate_rate`, `answer_rate`, `activation_to_routing_correlation`.

Plus a figure: **scatter plot of declared MSV activation (x-axis) vs routing action (y-axis, coded as ANSWER=0, DELIBERATE=1, DELEGATE=2), one subplot per model**. Reasoning models should show a flat horizontal line at ANSWER=0 regardless of activation. Non-reasoning models should show a positive slope.

**Implementation sketch:**
```python
def coherence_per_model(df):
    # Encode routing action as ordered integer
    action_code = {'ANSWER': 0, 'DELIBERATE': 1, 'DELEGATE': 2}
    df = df.copy()
    df['action_int'] = df['routing_action'].map(action_code)
    # Correlate declared activation with chosen action
    corr = df[['declared_activation', 'action_int']].corr().iloc[0, 1]
    return corr
```

**What we expect to see:** correlation between declared activation and chosen routing action is near zero for reasoning models and positive for non-reasoning models. This establishes H3: reasoning models produce varied self-reports that do not translate into varied routing behavior.

---

### Script 5: Cross-task convergence matrix (H1 + H2 + H3 combined)

Build a single summary matrix combining signals from Tasks 1, 2, and 11 to show the cross-task convergence.

**Inputs:** all three CSVs per model.
**Outputs:** `convergence_matrix.csv` with one row per model and columns:
- `model`
- `t01_delegate_slope` (slope of delegation rate vs difficulty)
- `t02_activation_to_routing_corr` (from Script 4)
- `t11_d_hat` (from Script 2)
- `t11_type2_auc` (from Script 2)
- `t11_mc` (from Script 2)
- `is_reasoning_model` (hand-labeled 0/1)

Plus a figure: **heatmap** of all these signals, rows sorted by `is_reasoning_model`, with reasoning models at the top. A clean pattern would show the reasoning-model rows with low `t01_delegate_slope`, low `t02_activation_to_routing_corr`, and low `t11_mc`, while non-reasoning-model rows show the opposite.

**What we expect to see:** if the pattern is real, the heatmap's reasoning-model block will be visually distinct from its non-reasoning-model block. That's the single most persuasive figure for the NeurIPS paper — it makes "reasoning-model metacognitive inefficiency" immediately legible without requiring the reader to follow the SDT derivation.

---

### Script 6: Verbosity-to-inefficiency correlation (H4)

Cross-reference Script 1's per-model verbosity statistics with Script 5's convergence-matrix metrics. Do models that emit more CoT tokens also show lower MC?

**Inputs:** `verbosity_stats.csv` and `convergence_matrix.csv`.
**Outputs:** `verbosity_vs_mc.csv` and a scatter plot of `output_tokens_mean` (x-axis) vs `mc` (y-axis), one point per model.

**What we expect to see:** if reasoning-training-induced verbosity is *the same underlying cause* as metacognitive inefficiency, the scatter should show a negative correlation: more CoT tokens, lower MC. If they dissociate (e.g., a verbose model with non-zero MC, or a terse model with zero MC), then the two are separate phenomena and the paper should say so.

---

## 4. Proposed writing plan once scripts run

Once Scripts 1–6 produce outputs, the NeurIPS paper's Results section can be structured as:

1. **Verbosity as a behavioral signature** (Script 1 figure). Establishes that "reasoning model" is a measurable category on this benchmark.
2. **Object-level vs metacognitive discrimination on Task 11** (Script 2 figure). Establishes H1 with bootstrap CIs.
3. **Flat delegation curves on Task 1** (Script 3 figure). Establishes H2.
4. **Declared-vs-behavioral decoupling on Task 2** (Script 4 figure). Establishes H3.
5. **Cross-task convergence** (Script 5 heatmap). Integrates all three signals into one legible visualization.
6. **Relationship between verbosity and efficiency** (Script 6 scatter). Tests whether these are one phenomenon or two.

This ordering goes from observable (output tokens) to derived (MC ratio) to synthetic (heatmap), which mirrors how readers will build their own mental model.

---

## 5. Additional data collection worth considering

Before NeurIPS submission, a few additional Kaggle runs would substantially strengthen the finding:

1. **Non-reasoning controls from the same model families** where available. If Zhipu has a non-reasoning GLM variant hosted on Kaggle, run it on Tasks 1 and 11 to show that the within-family contrast (reasoning vs non-reasoning) reproduces the pattern. Same for DeepSeek if V3.2 becomes available without reasoning mode.
2. **One additional reasoning model** not yet tested. Qwen3-235B-A22B-Thinking or o-series if accessible. A third data point in the "high $\hat{d}$, chance-level type-2 AUC" cluster would move the finding from two-model observation to three-model pattern.
3. **Task 1 runs with difficulty made explicit in the prompt**. One hypothesis is that reasoning models refuse to delegate because they don't perceive the difficulty of the current question. If explicitly told "this question has difficulty 0.85," does delegation rate increase? If yes, the failure is perception-of-difficulty; if no, the failure is motivational or architectural.

None of these are required for the current finding, but any one of them would let the NeurIPS paper move from "we observed this pattern in two models" to "we observed this pattern robustly across N models with a plausible causal story."

---

## 6. Implementation priority

If time is limited before the NeurIPS deadline, the minimum-viable pipeline is Scripts 1, 2, 3, and 5. These four scripts together produce enough evidence to write the section drafted above with confidence. Scripts 4 and 6 strengthen the story but are secondary.

Each of the six scripts should take under a day to implement and debug, with Scripts 1 and 3 being the simplest (pure data aggregation and plotting) and Scripts 2 and 5 being the most involved (bootstrap CIs and multi-source integration). A reasonable schedule for one developer is:

- Day 1: Scripts 0 and 1 (catalog plus verbosity analysis).
- Day 2: Scripts 2 and 3 (Task 11 metacognitive efficiency and Task 1 delegation curves).
- Day 3: Scripts 4 and 5 (Task 2 coherence and cross-task heatmap).
- Day 4: Script 6 and write-up of all figures and tables for the paper.

All scripts should produce standalone output files (CSVs + figures) that can be inspected and verified without re-running. This makes the analysis reproducible by coauthors and robust to last-minute changes in the model set.
