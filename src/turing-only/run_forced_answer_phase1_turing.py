#!/usr/bin/env python3
"""
run_forced_answer_phase1_turing.py
===================================

Runs the forced-answer Phase 1 protocol on the Turing HPC cluster for the
9 open-weight models. This produces declarative confidence data for all 80
GPQA Diamond items from every model, eliminating the answered-only bias in
the current Section 5.2 comparative analysis.

Purpose
-------
The paper's declarative metrics (ECE, Brier) are currently computed on
answered-only trials because the Phase 2 Delegate Game lets the model
delegate hard items, meaning no confidence is recorded for those items.
Phase 1 forces every model to answer every question with a 1-4 confidence
rating, producing a declarative-signal dataset aligned item-for-item with
Phase 2 behavioral outcomes.

How the Phase 2 comparative analysis changes after this:
  Before: ECE and Brier computed on answered-only subset (biased toward
          easier items for delegating models; especially unreliable for
          models with >90% delegation rates like gemma2:2b, llama3.2:3b,
          qwen2.5:3b, which had no answered-only signal at all).
  After:  ECE and Brier computed on the full 80-question set using Phase 1
          forced-answer confidence as the declarative signal. The comparative
          table retains all 9 models including those with 100% delegation
          in Phase 2, since they still produce declarative data in Phase 1.

Inputs
------
  --gpqa-jsonl    JSONL file with 80 GPQA Diamond questions used in the paper.
                  Expected schema: question_id, question, option_a, option_b,
                  option_c, option_d, correct_answer (letter A/B/C/D),
                  category, difficulty (0-1 empirical).
  --models        Comma-separated list of ollama model tags to run.
                  Default: the paper's 9-model cohort.
  --output-dir    Where to write per-model forced-answer CSVs.
  --seed          Random seed (default 0, deterministic decoding).

Outputs
-------
  <output-dir>/<model>.csv  Per-model file with columns:
      question_id, answer, confidence, correct, raw_response

Usage
-----
Single-model quick test (runs in ~3 minutes on Turing):

    python run_forced_answer_phase1_turing.py \\
        --gpqa-jsonl   ./gpqa_diamond_80.jsonl \\
        --models       llama3.2:1b \\
        --output-dir   ./results/forced_answer_phase1/ \\
        --seed         0

Full 9-model run (~2 hours total, sequential):

    python run_forced_answer_phase1_turing.py \\
        --gpqa-jsonl   ./gpqa_diamond_80.jsonl \\
        --output-dir   ./results/forced_answer_phase1/

Dependencies
------------
  ollama (server running locally), requests, pandas

Notes
-----
- Uses deterministic decoding (temperature=0, seed=0) for reproducibility.
  This matches the Turing Phase 2 decoding config.
- On parse failure, retries once with a more explicit prompt. On second
  failure, records parse_failure=True but keeps the trial (for completion
  accounting; the downstream analysis drops parse failures for ECE/Brier
  but uses them for completion rate).
- Does NOT hit the Kaggle API. This script is for Turing only.
- Uses ollama's /api/generate endpoint. If you're running a different
  local LLM server, adjust call_model() accordingly.
"""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests


DEFAULT_MODELS = [
    "qwen2.5:7b", "qwen2.5:3b",
    "llama3.1:8b", "llama3.2:3b", "llama3.2:1b",
    "phi4-mini",
    "gemma2:9b", "gemma2:2b",
    "mistral:7b",
]

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"


def build_prompt(row: dict) -> str:
    """Construct the forced-answer prompt for a single question.

    This prompt is deliberately minimal and leaves NO room for delegation.
    The model must commit to a letter and a 1-4 confidence.
    """
    opts = (
        f"A) {row['option_a']}\n"
        f"B) {row['option_b']}\n"
        f"C) {row['option_c']}\n"
        f"D) {row['option_d']}"
    )
    return (
        f"{row['question']}\n{opts}\n\n"
        "Answer this multiple-choice question. Choose A, B, C, or D and rate "
        "your confidence on a 1-4 scale "
        "(1=guessing, 2=uncertain, 3=fairly sure, 4=very confident).\n\n"
        "Think briefly if needed, then end your response with a single line "
        "of JSON in this format:\n"
        '{"answer": "<letter>", "confidence": <1-4>}\n\n'
        "Replace <letter> with your chosen A, B, C, or D, and replace <1-4> "
        "with an integer from 1 to 4."
    )


RETRY_PROMPT_SUFFIX = (
    "\n\nReminder: end your response with a single line of JSON in this exact "
    'form: {"answer": "<letter>", "confidence": <1-4>}, where <letter> is '
    'A/B/C/D and <1-4> is an integer from 1 to 4.'
)


def call_model(model: str, prompt: str, seed: int = 0,
               timeout_s: int = 90) -> str | None:
    """Call the local ollama server with deterministic settings."""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "top_p":       1.0,
                    "num_predict": 768,
                    "seed":        seed,
                },
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
    except Exception as e:
        print(f"    [api error] {e}")
        return None


def parse_response(text: str) -> tuple[str | None, int | None]:
    """Find the last JSON object in the output and extract answer/confidence.

    Searching backward matches our Phase 2 parser and handles models that
    produce reasoning before the structured answer.
    """
    if not text:
        return (None, None)
    matches = list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))
    for m in reversed(matches):
        try:
            obj = json.loads(m.group(0))
            ans = obj.get("answer", None)
            conf = obj.get("confidence", None)
            if ans in {"A", "B", "C", "D"} and isinstance(conf, (int, float)):
                conf_int = int(conf)
                if 1 <= conf_int <= 4:
                    return (ans, conf_int)
        except Exception:
            continue
    return (None, None)


def run_one_model(model: str, questions: list[dict], seed: int,
                  output_dir: Path) -> None:
    """Run Phase 1 over all 80 questions for a single model."""
    out_path = output_dir / f"{model.replace(':', '_').replace('/', '_')}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t0 = time.time()
    for i, q in enumerate(questions):
        prompt = build_prompt(q)
        resp = call_model(model, prompt, seed=seed)
        ans, conf = parse_response(resp or "")

        # Retry once if parse failure
        if ans is None or conf is None:
            resp2 = call_model(model, prompt + RETRY_PROMPT_SUFFIX, seed=seed)
            ans, conf = parse_response(resp2 or "")
            raw_response = (resp or "") + "\n\n<<RETRY>>\n\n" + (resp2 or "")
        else:
            raw_response = resp or ""

        correct_letter = q.get("correct_answer", "").strip().upper()
        rows.append({
            "question_id": q["question_id"],
            "answer":      ans if ans is not None else "",
            "confidence":  conf if conf is not None else "",
            "correct":     1 if (ans == correct_letter) else 0,
            "raw_response": raw_response[:2000],
            "parse_failure": ans is None or conf is None,
        })
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(questions)} questions  "
                  f"({time.time() - t0:.0f}s elapsed)")

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  Saved: {out_path}  ({time.time() - t0:.0f}s total)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gpqa-jsonl", type=Path, required=True,
                    help="JSONL file with 80 GPQA Diamond questions")
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS),
                    help="Comma-separated ollama model tags "
                         "(default: 9-model cohort)")
    ap.add_argument("--output-dir", type=Path,
                    default=Path("./results/forced_answer_phase1/"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with args.gpqa_jsonl.open() as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} GPQA Diamond questions")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Running {len(models)} models: {models}")

    for model in models:
        print(f"\n=== {model} ===")
        run_one_model(model, questions, args.seed, args.output_dir)

    print("\nAll models complete.")
    print(f"Output: {args.output_dir}/")
    print("\nNext step: feed these CSVs to compute_bootstrap_ci.py via "
          "--forced_answer_dir to replace the answered-only declarative "
          "baseline in Table 3.")


if __name__ == "__main__":
    main()
