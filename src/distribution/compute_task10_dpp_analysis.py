#!/usr/bin/env python3
"""
compute_task10_dpp_analysis.py

Task 10 DPP institutional analysis -- reproduces every table and number in
paper Appendix `app:task10_dpp_institutional`.

Inputs:
  --dpp-dir           Directory of per-model DPP CSVs and __transcripts/ subdirs
                      (typically distribution-msv-benchmark/data/task10_dpp/)
  --fa-dir            Directory of per-model Forced-Answer Phase 1 CSVs
                      (typically distribution-msv-benchmark/data/forced_answer_phase1/)
  --difficulty-csv    Optional. gpqa_difficulty_scores.csv. Used only as a
                      consistency probe (paper-bound results do not stratify
                      by difficulty -- the 80-question panel is too clumped
                      at max difficulty to give useful tertile bins).
  --output-dir        Where the four output CSVs go.
  --n-boot            Bootstrap iterations for paired-bootstrap CIs.
                      Default 10000. Use lower for development.
  --seed              Bootstrap RNG seed. Default 42.

Outputs (under --output-dir):
  task10_lift.csv          Per-model end-to-end and matched-paired lift with
                           95% paired-bootstrap CIs. Cohort-aggregate row.
  task10_winloss.csv       Per-model paired wins/losses, churn, McNemar exact p.
                           Cohort-aggregate row.
  task10_trace.csv         Per-model Expert->Generalist trace correction:
                           rescue/harm/stable_correct/stable_wrong counts plus
                           binomial p on (rescues > harms).
  task10_extract_qc.csv    Per-model Expert-letter extraction success rate
                           (the layered-regex extractor's coverage). Models
                           with extraction-rate < 0.80 get a flag column.

Method summary (paper Section 5 + Appendix `app:task10_dpp_institutional`):

  Lift = mean(DPP_correct) - mean(FA_correct), restricted to the FA-parsed
  subset for each model (FA's parse_failure==0 rows). The FA-parsed subset
  varies by model: gemma2:2b 80/80, most models 73-78/80, llama3.2:1b 64/80,
  phi4-mini 48/80.

  Paired-bootstrap CI: B=10000 resamples of the question-level diff vector
  d = DPP_correct - FA_correct. 2.5/97.5 percentiles. Seed default 42.

  Win/loss: for each question in the matched set, classify as
    win   = DPP correct, FA wrong   (1, 0)
    loss  = DPP wrong, FA correct   (0, 1)
    tie_correct  = both correct     (1, 1)
    tie_wrong    = both wrong       (0, 0)
  Discordant = wins + losses. Churn rate = discordant / matched_n.
  McNemar exact: scipy.stats.binomtest(min(wins, losses), wins+losses, p=0.5).

  Expert->Generalist trace: parse Expert-stage final letter from
  transcripts/<question_id>.json via layered regex. Compare against
  Generalist (= final_letter from CSV). Categorize each item:
    rescue       = Expert wrong, Final correct
    harm         = Expert correct, Final wrong
    stable_correct = both correct
    stable_wrong = both wrong
  Binomial test on rescues vs harms (two-sided, matches McNemar convention):
  scipy.stats.binomtest(rescues, rescues+harms, p=0.5, alternative='two-sided').

  Models with Expert extraction-rate < 0.80 are flagged with a `low_extract`
  column in task10_trace.csv. The paper excludes their trace numbers from
  prose (e.g. llama3.2:1b at ~66%) but retains them in the table with the
  flag column for transparency.

Worked example numbers (qwen2.5:7b, FA-parsed n=78):
  end-to-end lift = 0.4625 - 0.4625 = +0.0000
  matched lift    = 0.474 - 0.449 = +0.026, paired-bootstrap 95% CI ~[-0.064, +0.115]
  wins=14, losses=12, discordant=26, churn=0.333, McNemar p=0.84
  Expert acc 0.564, Final acc 0.474, rescues=11, harms=18, net=-7, binom p=0.84

Worked example numbers (llama3.1:8b, FA-parsed n=73):
  matched lift    = +0.123, paired-bootstrap 95% CI ~[+0.027, +0.219]
  wins=18, losses=9, discordant=27, churn=0.370, McNemar exact p=0.108
  Expert acc 0.380, Final acc 0.494, rescues=16, harms=7, net=+9, binom p=0.093

Cohort aggregate (all 9 models, all matched pairs pooled):
  total wins=120, total losses=109, total discordant=229, total matched=648,
  cohort churn=0.354, cohort mean lift=+0.021
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# Layered regex patterns for Expert-stage final-letter extraction. Order
# matters: more specific patterns first, fallbacks last. Each pattern is a
# (regex, name) pair where `name` is reported in the QC CSV.
#
# Patterns derived from inspecting expert-stage outputs across all 9 models;
# captures formats observed in:
#   qwen2.5:7b      "Answer: C", bare letter at start ("A\n\n")
#   qwen2.5:3b      "**A/B/C/D: D**", "The correct answer is B)"
#   llama3.1:8b     "A)", "A\n\n"
#   llama3.2:1b     "would recommend option A)", "answer is A"
#   llama3.2:3b     "The correct answer is C)", "(B)"
#   mistral:7b      free-form prose, often no explicit letter
#   phi4-mini       "C) Zn in ether..."
#   gemma2:2b       "The correct answer is **B)..."
#   gemma2:9b       "**Answer: A)...", "The answer is **(A)**"
EXPERT_PATTERNS: list[tuple[str, str]] = [
    # 1. Explicit Answer: X tag (most reliable; many models)
    (r"^\s*\*{0,2}\s*Answer\s*:\s*\*{0,2}\s*\(?([A-D])\)?", "Answer: X"),
    # 2. JSON-style {"answer": "X"}
    (r'\{[^}]*"answer"\s*:\s*"([A-D])"', 'JSON {"answer": "X"}'),
    # 3. "**A/B/C/D: D**" or "**A/B/C/D**: A" enumerator pattern (qwen2.5:3b)
    (r"\*\*\s*A\s*/\s*B\s*/\s*C\s*/\s*D\s*\*?\*?\s*:\s*\*?\*?\s*([A-D])\s*\*{0,2}", "**A/B/C/D**: X"),
    # 4. "The (correct) answer is X" with optional formatting
    (
        r"\b[Tt]he\s+(?:correct\s+)?answer\s+is\s*\*{0,2}\s*\(?([A-D])\)?",
        "the answer is X",
    ),
    # 5. "Final answer/choice: X"
    (
        r"\bFinal\s+(?:answer|choice)\s*[:=]\s*\*{0,2}\s*\(?([A-D])\)?",
        "Final answer: X",
    ),
    # 6. "The correct choice/option is X" / "recommend option X"
    (
        r"\b(?:recommend|correct\s+(?:choice|option))\s+(?:is\s+)?(?:option\s+)?\(?([A-D])\)?",
        "correct option/choice X",
    ),
    # 7. "The correct sequence is **C) ..." (gemma2 pattern)
    (
        r"^\s*[Tt]he\s+correct\s+\w+\s+is\s+\*{0,2}\s*\(?([A-D])\)?",
        "the correct <noun> is X",
    ),
    # 8. Bare letter then ")" or "." or whitespace at start of response
    #    Example: "A)\n", "C\n\n", "B."
    (r"^\s*\*{0,2}\s*\(?([A-D])\)\s*[\.\n\s]", "X) at start"),
    (r"^\s*([A-D])\s*\n", "X alone at start of line"),
    # 9. Bracketed / asterisk-wrapped letter near the start
    (r"^[^A-Za-z]{0,40}\(([A-D])\)", "(X) near start"),
    (r"^[^A-Za-z]{0,40}\*\*\s*\(?([A-D])\)?\s*\*\*", "**X** near start"),
    # 10. Last-resort fallback: any standalone (X) in first 200 chars
    (r"\(([A-D])\)\s*[\s\n]", "(X) anywhere in first 200 chars"),
]


GENERALIST_PATTERNS: list[tuple[str, str]] = [
    # Generalist responses are nearly always JSON {"answer": "X"} but there
    # are stray whitespace cases. We use the same patterns as Expert as a
    # fallback in case the JSON parse fails.
    (r'\{[^}]*"answer"\s*:\s*"([A-D])"', 'JSON {"answer": "X"}'),
    (r'^\s*\(?([A-D])\)?\s*$', "bare X"),
    (r'\b[Tt]he\s+answer\s+is\s+\(?([A-D])\)?', "the answer is X"),
]


def extract_letter(
    text: str,
    patterns: list[tuple[str, str]],
    head_chars: int = 200,
) -> tuple[Optional[str], str]:
    """
    Apply layered regex patterns in order, return the first match.

    Patterns 1-7 (specific) match anywhere; patterns 8-10 (positional)
    match near the start. We restrict patterns 8-10 by truncating to the
    first `head_chars` characters before applying them.

    Returns (letter or None, name_of_pattern_that_matched_or_'unmatched').
    """
    if not text:
        return None, "empty"
    text = text.strip()
    head = text[:head_chars]
    # Try each pattern in declared order, applying anchored ones to head only
    for i, (pat, name) in enumerate(patterns):
        target = head if (pat.startswith("^") or "near start" in name or "at start" in name) else text
        m = re.search(pat, target, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).upper(), name
    return None, "unmatched"


def paired_bootstrap_ci(
    diffs: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float]:
    """
    Paired-item bootstrap CI on mean(diffs).

    `diffs` is the question-level vector DPP_correct - FA_correct restricted
    to questions where both panels produced a parseable answer. Resampling
    is paired (i.e. by question index, not separately), which is the
    appropriate procedure for matched-pair lift.

    Returns (point_estimate, ci_low, ci_high) at 2.5/97.5 percentiles.
    """
    rng = np.random.default_rng(seed)
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # Vectorized resampling for speed
    idx = rng.integers(0, n, size=(n_boot, n))
    means = diffs[idx].mean(axis=1)
    return float(diffs.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def mcnemar_exact(wins: int, losses: int) -> float:
    """Exact McNemar p-value via two-sided binomial on min(wins, losses)."""
    n = wins + losses
    if n == 0:
        return float("nan")
    k = min(wins, losses)
    # Two-sided binomial: P(X <= k) + P(X >= n-k) for X ~ Binomial(n, 0.5)
    return float(binomtest(k, n, p=0.5, alternative="two-sided").pvalue)


def binomial_two_sided(rescues: int, harms: int) -> float:
    """
    Two-sided binomial p-value on (rescues vs harms) under H0: p=0.5.

    Paper convention is two-sided (matches McNemar exact convention).
    For per-model rescue/harm counts the two-sided value is what the
    paper reports.
    """
    n = rescues + harms
    if n == 0:
        return float("nan")
    return float(binomtest(rescues, n, p=0.5, alternative="two-sided").pvalue)


def normalize_model_name(filename: str) -> str:
    """
    Strip .csv suffix; preserve underscores and dots (the file naming
    convention is e.g. 'qwen2.5_7b.csv' -> 'qwen2.5_7b').
    """
    return Path(filename).stem


def load_dpp_csv(path: Path) -> pd.DataFrame:
    """Load DPP CSV; ensure question_id is the index for paired ops."""
    df = pd.read_csv(path)
    # CSV writes question_id as first column; some models have CR endings
    df.columns = [c.strip() for c in df.columns]
    # Cast `correct` to int (some files may have it as bool)
    df["correct"] = df["correct"].astype(int)
    return df


def load_fa_csv(path: Path) -> pd.DataFrame:
    """Load FA Phase 1 CSV; mark parse-failure rows."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["correct"] = df["correct"].astype(int)
    df["parse_failure"] = df["parse_failure"].astype(int)
    return df


def parse_expert_letters_for_model(
    transcripts_dir: Path,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Iterate transcripts/<question_id>.json for one model. For each item,
    extract Expert-stage final letter and Generalist-stage final letter.

    Returns:
      df with columns: question_id, expert_letter, generalist_letter,
                       expert_pattern, generalist_pattern
      summary dict of pattern-hit counts
    """
    rows = []
    pattern_counts: dict[str, int] = {}
    if not transcripts_dir.exists():
        return pd.DataFrame(columns=[
            "question_id", "expert_letter", "generalist_letter",
            "expert_pattern", "generalist_pattern",
        ]), pattern_counts

    json_files = sorted(transcripts_dir.glob("*.json"))
    for jf in json_files:
        qid = jf.stem
        try:
            with open(jf) as f:
                t = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        transcript = t.get("transcript", {})
        expert_text = transcript.get("expert", "") or ""
        generalist_text = transcript.get("generalist", "") or ""

        e_letter, e_pat = extract_letter(expert_text, EXPERT_PATTERNS)
        g_letter, g_pat = extract_letter(generalist_text, GENERALIST_PATTERNS)
        pattern_counts[e_pat] = pattern_counts.get(e_pat, 0) + 1

        rows.append({
            "question_id": qid,
            "expert_letter": e_letter,
            "generalist_letter": g_letter,
            "expert_pattern": e_pat,
            "generalist_pattern": g_pat,
        })

    df = pd.DataFrame(rows)
    return df, pattern_counts


def analyze_one_model(
    model: str,
    dpp_csv: Path,
    fa_csv: Path,
    transcripts_dir: Path,
    n_boot: int,
    seed: int,
) -> tuple[dict, dict, dict, dict]:
    """
    Returns (lift_row, winloss_row, trace_row, qc_row) for one model.
    """
    dpp = load_dpp_csv(dpp_csv)
    fa = load_fa_csv(fa_csv)

    # End-to-end lift: cohort-level mean over the full panels (no matching)
    end_to_end_lift = float(dpp["correct"].mean() - fa["correct"].mean())

    # Matched subset: FA-parsed (parse_failure == 0) inner-join on question_id
    fa_parsed = fa[fa["parse_failure"] == 0]
    merged = pd.merge(
        dpp[["question_id", "correct"]].rename(columns={"correct": "dpp_correct"}),
        fa_parsed[["question_id", "correct"]].rename(columns={"correct": "fa_correct"}),
        on="question_id",
        how="inner",
    )
    n_matched = len(merged)
    n_fa_total = len(fa)
    n_fa_parsed = len(fa_parsed)
    fa_parse_rate = n_fa_parsed / n_fa_total if n_fa_total else float("nan")

    # Matched-paired lift with bootstrap CI
    diffs = (merged["dpp_correct"] - merged["fa_correct"]).to_numpy(dtype=float)
    matched_lift, ci_low, ci_high = paired_bootstrap_ci(diffs, n_boot=n_boot, seed=seed)

    # Win/loss decomposition
    wins = int(((merged["dpp_correct"] == 1) & (merged["fa_correct"] == 0)).sum())
    losses = int(((merged["dpp_correct"] == 0) & (merged["fa_correct"] == 1)).sum())
    tie_correct = int(((merged["dpp_correct"] == 1) & (merged["fa_correct"] == 1)).sum())
    tie_wrong = int(((merged["dpp_correct"] == 0) & (merged["fa_correct"] == 0)).sum())
    discordant = wins + losses
    churn = discordant / n_matched if n_matched else float("nan")
    p_mcnemar = mcnemar_exact(wins, losses)

    # Expert -> Generalist trace correction
    trace_df, pattern_counts = parse_expert_letters_for_model(transcripts_dir)

    # Join trace with DPP results for ground-truth `correct_letter`
    if not trace_df.empty:
        trace_full = pd.merge(
            trace_df,
            dpp[["question_id", "correct_letter", "final_letter"]],
            on="question_id",
            how="inner",
        )
        # Expert correctness from extracted letter (None -> not-counted)
        trace_full["expert_correct"] = (
            trace_full["expert_letter"].notna()
            & (trace_full["expert_letter"] == trace_full["correct_letter"])
        ).astype("Int64")
        trace_full.loc[trace_full["expert_letter"].isna(), "expert_correct"] = pd.NA
        # Final correctness from CSV `correct` column for the same questions
        trace_full = pd.merge(
            trace_full,
            dpp[["question_id", "correct"]].rename(columns={"correct": "final_correct"}),
            on="question_id",
            how="inner",
        )
        # Categorize each item where Expert was extractable
        valid = trace_full[trace_full["expert_correct"].notna()].copy()
        rescues = int(((valid["expert_correct"] == 0) & (valid["final_correct"] == 1)).sum())
        harms = int(((valid["expert_correct"] == 1) & (valid["final_correct"] == 0)).sum())
        stable_correct = int(((valid["expert_correct"] == 1) & (valid["final_correct"] == 1)).sum())
        stable_wrong = int(((valid["expert_correct"] == 0) & (valid["final_correct"] == 0)).sum())
        n_extracted = len(valid)
        n_total_transcripts = len(trace_full)
        extract_rate = n_extracted / n_total_transcripts if n_total_transcripts else float("nan")
        expert_acc = float(valid["expert_correct"].mean()) if n_extracted else float("nan")
        final_acc_on_extracted = float(valid["final_correct"].mean()) if n_extracted else float("nan")
        binom_p = binomial_two_sided(rescues, harms)
    else:
        rescues = harms = stable_correct = stable_wrong = 0
        n_extracted = n_total_transcripts = 0
        extract_rate = expert_acc = final_acc_on_extracted = float("nan")
        binom_p = float("nan")

    low_extract = int(extract_rate < 0.80) if not np.isnan(extract_rate) else 1

    lift_row = {
        "model": model,
        "n_dpp": len(dpp),
        "n_fa_total": n_fa_total,
        "n_fa_parsed": n_fa_parsed,
        "fa_parse_rate": round(fa_parse_rate, 4),
        "n_matched": n_matched,
        "dpp_acc": round(float(dpp["correct"].mean()), 4),
        "fa_acc_full": round(float(fa["correct"].mean()), 4),
        "fa_acc_parsed": round(float(fa_parsed["correct"].mean()), 4) if n_fa_parsed else float("nan"),
        "end_to_end_lift": round(end_to_end_lift, 4),
        "matched_lift": round(matched_lift, 4),
        "matched_lift_ci_low": round(ci_low, 4),
        "matched_lift_ci_high": round(ci_high, 4),
        "n_boot": n_boot,
        "ci_excludes_zero": int((ci_low > 0) or (ci_high < 0)),
    }

    winloss_row = {
        "model": model,
        "n_matched": n_matched,
        "wins": wins,
        "losses": losses,
        "tie_correct": tie_correct,
        "tie_wrong": tie_wrong,
        "discordant": discordant,
        "churn_rate": round(churn, 4),
        "mcnemar_exact_p": round(p_mcnemar, 4),
        "mcnemar_significant_05": int(p_mcnemar < 0.05) if not np.isnan(p_mcnemar) else 0,
    }

    trace_row = {
        "model": model,
        "n_total_transcripts": n_total_transcripts,
        "n_expert_extracted": n_extracted,
        "expert_extract_rate": round(extract_rate, 4) if not np.isnan(extract_rate) else float("nan"),
        "low_extract": low_extract,
        "expert_acc": round(expert_acc, 4) if not np.isnan(expert_acc) else float("nan"),
        "final_acc_on_extracted": round(final_acc_on_extracted, 4) if not np.isnan(final_acc_on_extracted) else float("nan"),
        "rescues": rescues,
        "harms": harms,
        "net_correction": rescues - harms,
        "stable_correct": stable_correct,
        "stable_wrong": stable_wrong,
        "binomial_two_sided_p": round(binom_p, 4) if not np.isnan(binom_p) else float("nan"),
    }

    qc_row = {
        "model": model,
        "n_total_transcripts": n_total_transcripts,
        "n_expert_extracted": n_extracted,
        "expert_extract_rate": round(extract_rate, 4) if not np.isnan(extract_rate) else float("nan"),
        "low_extract_flag": low_extract,
        "top_pattern_hits": "; ".join(
            f"{name}={count}"
            for name, count in sorted(pattern_counts.items(), key=lambda kv: -kv[1])[:5]
        ),
    }

    return lift_row, winloss_row, trace_row, qc_row


def cohort_aggregate(
    per_model_lifts: list[dict],
    per_model_winlosses: list[dict],
    n_boot: int,
    seed: int,
    all_diffs: np.ndarray,
) -> tuple[list[dict], dict]:
    """
    Two cohort-aggregate rows for the lift table:

    1. COHORT_MEAN_OF_MODELS: unweighted mean of per-model matched lifts.
       This is the paper convention -- each model contributes equally
       regardless of its FA-parsed n. Paper-locked value: +0.021.

    2. COHORT_POOLED: pooled-diff bootstrap on the concatenated
       question-level diff vector. Larger-n models contribute more.
       Defensible alternative aggregate; useful for sensitivity reporting.

    Win/loss aggregate is a single COHORT_AGGREGATE row -- simple sum
    across models. Paper-locked: 120 wins, 109 losses, 229 discordant,
    648 matched, churn = 0.354.
    """
    # 1. Unweighted mean of per-model lifts (paper convention)
    per_model_lift_values = np.array([r["matched_lift"] for r in per_model_lifts], dtype=float)
    per_model_n = np.array([r["n_matched"] for r in per_model_lifts], dtype=int)
    unweighted_mean = float(per_model_lift_values.mean()) if len(per_model_lift_values) else float("nan")
    # CI: bootstrap over models (resample which models go into the cohort)
    rng = np.random.default_rng(seed + 1)  # different seed stream from per-model bootstrap
    M = len(per_model_lift_values)
    if M:
        boot_means = np.array([
            per_model_lift_values[rng.integers(0, M, size=M)].mean()
            for _ in range(n_boot)
        ])
        unweighted_ci_low = float(np.percentile(boot_means, 2.5))
        unweighted_ci_high = float(np.percentile(boot_means, 97.5))
    else:
        unweighted_ci_low = unweighted_ci_high = float("nan")

    # 2. Pooled-diff bootstrap (n-weighted)
    pooled_lift, pooled_ci_low, pooled_ci_high = paired_bootstrap_ci(
        all_diffs, n_boot=n_boot, seed=seed
    )

    cohort_lift_unweighted = {
        "model": "COHORT_MEAN_OF_MODELS",
        "n_dpp": "",
        "n_fa_total": "",
        "n_fa_parsed": "",
        "fa_parse_rate": "",
        "n_matched": int(per_model_n.sum()),
        "dpp_acc": "",
        "fa_acc_full": "",
        "fa_acc_parsed": "",
        "end_to_end_lift": "",
        "matched_lift": round(unweighted_mean, 4),
        "matched_lift_ci_low": round(unweighted_ci_low, 4),
        "matched_lift_ci_high": round(unweighted_ci_high, 4),
        "n_boot": n_boot,
        "ci_excludes_zero": int((unweighted_ci_low > 0) or (unweighted_ci_high < 0)),
    }
    cohort_lift_pooled = {
        "model": "COHORT_POOLED",
        "n_dpp": "",
        "n_fa_total": "",
        "n_fa_parsed": "",
        "fa_parse_rate": "",
        "n_matched": len(all_diffs),
        "dpp_acc": "",
        "fa_acc_full": "",
        "fa_acc_parsed": "",
        "end_to_end_lift": "",
        "matched_lift": round(pooled_lift, 4),
        "matched_lift_ci_low": round(pooled_ci_low, 4),
        "matched_lift_ci_high": round(pooled_ci_high, 4),
        "n_boot": n_boot,
        "ci_excludes_zero": int((pooled_ci_low > 0) or (pooled_ci_high < 0)),
    }

    cohort_n_matched = sum(r["n_matched"] for r in per_model_winlosses)
    cohort_wins = sum(r["wins"] for r in per_model_winlosses)
    cohort_losses = sum(r["losses"] for r in per_model_winlosses)
    cohort_tie_correct = sum(r["tie_correct"] for r in per_model_winlosses)
    cohort_tie_wrong = sum(r["tie_wrong"] for r in per_model_winlosses)
    cohort_discordant = cohort_wins + cohort_losses
    cohort_churn = cohort_discordant / cohort_n_matched if cohort_n_matched else float("nan")
    cohort_mcnemar = mcnemar_exact(cohort_wins, cohort_losses)

    cohort_winloss = {
        "model": "COHORT_AGGREGATE",
        "n_matched": cohort_n_matched,
        "wins": cohort_wins,
        "losses": cohort_losses,
        "tie_correct": cohort_tie_correct,
        "tie_wrong": cohort_tie_wrong,
        "discordant": cohort_discordant,
        "churn_rate": round(cohort_churn, 4),
        "mcnemar_exact_p": round(cohort_mcnemar, 4),
        "mcnemar_significant_05": int(cohort_mcnemar < 0.05) if not np.isnan(cohort_mcnemar) else 0,
    }
    return [cohort_lift_unweighted, cohort_lift_pooled], cohort_winloss


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dpp-dir", type=Path, required=True,
                    help="Directory of per-model Task 10 DPP CSVs and __transcripts/ subdirs")
    ap.add_argument("--fa-dir", type=Path, required=True,
                    help="Directory of per-model Forced-Answer Phase 1 CSVs")
    ap.add_argument("--difficulty-csv", type=Path, default=None,
                    help="Optional gpqa_difficulty_scores.csv (consistency probe only)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the four output CSVs")
    ap.add_argument("--n-boot", type=int, default=10000,
                    help="Bootstrap iterations (default 10000)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Bootstrap RNG seed (default 42)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover models from DPP CSV filenames; require matching FA CSV
    dpp_csvs = sorted(args.dpp_dir.glob("*.csv"))
    if not dpp_csvs:
        print(f"ERROR: no DPP CSVs found at {args.dpp_dir}")
        return 1

    per_model_lifts: list[dict] = []
    per_model_winlosses: list[dict] = []
    per_model_traces: list[dict] = []
    per_model_qc: list[dict] = []
    all_diffs_pool: list[np.ndarray] = []

    print(f">>> Analyzing {len(dpp_csvs)} models")
    print(f">>> n_boot={args.n_boot}, seed={args.seed}")
    print()

    for dpp_csv in dpp_csvs:
        model = normalize_model_name(dpp_csv.name)
        fa_csv = args.fa_dir / dpp_csv.name
        transcripts_dir = args.dpp_dir / f"{model}__transcripts"
        if not fa_csv.exists():
            print(f"  [{model}] WARNING: no matching FA CSV at {fa_csv}; skipping")
            continue

        print(f"  [{model}] processing...", flush=True)

        lift_row, winloss_row, trace_row, qc_row = analyze_one_model(
            model=model,
            dpp_csv=dpp_csv,
            fa_csv=fa_csv,
            transcripts_dir=transcripts_dir,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        per_model_lifts.append(lift_row)
        per_model_winlosses.append(winloss_row)
        per_model_traces.append(trace_row)
        per_model_qc.append(qc_row)

        # Recompute diffs for cohort pool (not stored on lift_row)
        dpp = load_dpp_csv(dpp_csv)
        fa = load_fa_csv(fa_csv)
        fa_parsed = fa[fa["parse_failure"] == 0]
        merged = pd.merge(
            dpp[["question_id", "correct"]].rename(columns={"correct": "dpp_correct"}),
            fa_parsed[["question_id", "correct"]].rename(columns={"correct": "fa_correct"}),
            on="question_id",
            how="inner",
        )
        all_diffs_pool.append((merged["dpp_correct"] - merged["fa_correct"]).to_numpy(dtype=float))

    # Cohort aggregate rows
    all_diffs = np.concatenate(all_diffs_pool) if all_diffs_pool else np.array([])
    cohort_lift_rows, cohort_winloss = cohort_aggregate(
        per_model_lifts, per_model_winlosses,
        n_boot=args.n_boot, seed=args.seed, all_diffs=all_diffs,
    )
    per_model_lifts.extend(cohort_lift_rows)
    per_model_winlosses.append(cohort_winloss)

    # Write outputs
    pd.DataFrame(per_model_lifts).to_csv(args.output_dir / "task10_lift.csv", index=False)
    pd.DataFrame(per_model_winlosses).to_csv(args.output_dir / "task10_winloss.csv", index=False)
    pd.DataFrame(per_model_traces).to_csv(args.output_dir / "task10_trace.csv", index=False)
    pd.DataFrame(per_model_qc).to_csv(args.output_dir / "task10_extract_qc.csv", index=False)

    print()
    print(f">>> Wrote 4 CSVs to {args.output_dir}/")
    print(f"    task10_lift.csv          ({len(per_model_lifts)} rows incl. 2 cohort aggregates)")
    print(f"    task10_winloss.csv       ({len(per_model_winlosses)} rows incl. cohort aggregate)")
    print(f"    task10_trace.csv         ({len(per_model_traces)} rows)")
    print(f"    task10_extract_qc.csv    ({len(per_model_qc)} rows)")
    print()
    unweighted = cohort_lift_rows[0]
    pooled = cohort_lift_rows[1]
    print(f">>> Cohort lift (paper convention, unweighted mean of models): "
          f"{unweighted['matched_lift']:+.4f} "
          f"95% CI [{unweighted['matched_lift_ci_low']:+.4f}, {unweighted['matched_lift_ci_high']:+.4f}]")
    print(f">>> Cohort lift (pooled-diff bootstrap, n-weighted): "
          f"{pooled['matched_lift']:+.4f} "
          f"95% CI [{pooled['matched_lift_ci_low']:+.4f}, {pooled['matched_lift_ci_high']:+.4f}]")
    print(f">>> Win/loss: wins={cohort_winloss['wins']}, losses={cohort_winloss['losses']}, "
          f"discordant={cohort_winloss['discordant']}, "
          f"churn={cohort_winloss['churn_rate']:.4f}, "
          f"McNemar exact p={cohort_winloss['mcnemar_exact_p']:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
