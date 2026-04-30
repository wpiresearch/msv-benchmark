#!/usr/bin/env python3
"""
run_task10_dpp_turing.py
=========================

Runs the Dialectical Deliberative Prompt (DPP) protocol on the Turing HPC
cluster for the 9 open-weight models, with extended context budgets to
avoid the near-universal context-length failures observed on the Kaggle
cohort (only 1 of 23 models completed Task 10 cleanly).

Protocol
--------
For each GPQA Diamond question, the model is called five times in sequence:

  1. Expert        - answer with full technical authority
  2. Critic        - identify weaknesses in the Expert's reasoning
  3. Evaluator     - assess which objections are valid
  4. Synthesizer   - produce the most defensible answer given all above
  5. Generalist    - state the final answer clearly

Each subsequent stage receives the full prior-stage output as context, so
by stage 5 the total context is Expert + Critic + Evaluator + Synthesizer +
new user turn, which can easily exceed 4-8k tokens.

What failed on Kaggle
---------------------
The Kaggle SDK passes a platform-default num_ctx that varies by model; for
most of the 23 Kaggle models the default is 4096 or 8192, and on hard GPQA
items the Expert and Critic responses alone often exceed the Synthesizer's
remaining input budget. 22 of 23 Kaggle models truncated partway through.

What this script does differently
---------------------------------
- Sets num_ctx=32768 per call (large enough for all five DPP stages on
  every model in the 9-model cohort that supports it).
- Logs per-stage token counts so you can see where each model sits.
- Tolerates stage failures (saves partial results; does not break the
  loop).
- Uses ollama's /api/generate endpoint, matching the Turing Phase 2 driver.

Inputs
------
  --gpqa-jsonl    Same 80-question JSONL used for Phase 1 and Phase 2.
  --models        Comma-separated model tags. Default: 9-model cohort.
  --output-dir    Where to write per-model DPP CSVs + raw transcripts.
  --num-ctx       Context window per call (default 32768).
  --seed          Random seed (default 0).
  --timeout-s     Per-call HTTP timeout (default 300s; DPP stages can be long).

Outputs
-------
  <output-dir>/<model>.csv             Per-question DPP result:
      question_id, expert_answer, critic_answer, evaluator_answer,
      synthesizer_answer, generalist_answer (the final letter A/B/C/D),
      correct_letter, correct, context_overflow (bool),
      stage_failures (int), elapsed_s

  <output-dir>/<model>__transcripts/   One JSON file per question with the
      full five-stage transcript, token counts per stage, and timing.

Usage
-----
Single-model test (llama3.1:8b, ~30 minutes sequential):

    python run_task10_dpp_turing.py \\
        --gpqa-jsonl ./gpqa_diamond_80.jsonl \\
        --models llama3.1:8b \\
        --output-dir ./results/task10_dpp/ \\
        --num-ctx 32768

Full 9-model run (~30-50 hours total, sequential):

    python run_task10_dpp_turing.py \\
        --gpqa-jsonl ./gpqa_diamond_80.jsonl \\
        --output-dir ./results/task10_dpp/ \\
        --num-ctx 32768

Dependencies
------------
  ollama (server running locally), requests, pandas
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


# Role prompts (identical to the Kaggle Task 10 notebook for apples-to-apples
# comparison).
ROLE_EXPERT = (
    "ROLE: You are a domain Expert. Answer the following question with full "
    "technical authority. State your answer (A/B/C/D) and your reasoning "
    "concisely."
)
ROLE_CRITIC = (
    "ROLE: You are a rigorous Critic. The Expert has just given an answer. "
    "Identify the weakest points in the Expert's reasoning. Be specific. "
    "Do not give your own answer yet; focus on critiquing."
)
ROLE_EVALUATOR = (
    "ROLE: You are an Evaluator. You have seen the Expert's answer and the "
    "Critic's objections. Assess which objections are valid and which are "
    "not. Be explicit about which parts of the Expert's reasoning survive "
    "the Critic's challenge."
)
ROLE_SYNTHESIZER = (
    "ROLE: You are a Synthesizer. Given the Expert answer, the Critic's "
    "objections, and the Evaluator's assessment, produce the most defensible "
    "final answer. State the letter (A/B/C/D) and a brief justification."
)
ROLE_GENERALIST = (
    "ROLE: You are a Generalist communicator. Given the full deliberation, "
    "state the final answer clearly. Select A, B, C, or D.\n"
    "Respond with ONLY a single line of JSON, nothing else.\n"
    '{"answer": "A"}'
)


def build_question_block(row: dict) -> str:
    return (
        f"{row['question']}\n"
        f"A) {row['option_a']}\n"
        f"B) {row['option_b']}\n"
        f"C) {row['option_c']}\n"
        f"D) {row['option_d']}"
    )


def call_ollama(model: str, prompt: str, seed: int, num_ctx: int,
                timeout_s: int) -> dict | None:
    """Call ollama and return the raw JSON payload (includes token counts)."""
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
                    "num_predict": 1024,
                    "num_ctx":     num_ctx,
                    "seed":        seed,
                },
            },
            timeout=timeout_s,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"    [api error] {e}")
        return None


def parse_letter_from_json(text: str) -> str | None:
    """Look backward through all JSON objects for {"answer": "A"}-shape."""
    if not text:
        return None
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
            if obj.get("answer") in {"A", "B", "C", "D"}:
                return obj["answer"]
        except Exception:
            continue
    # Fallback: find a bare letter in the last line
    last_line = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.search(r"\b([ABCD])\b", last_line)
    return m.group(1) if m else None


def run_dpp_one_question(model: str, row: dict, seed: int, num_ctx: int,
                         timeout_s: int) -> dict:
    """Run the 5-stage DPP pipeline on a single question."""
    q_block = build_question_block(row)

    # Stage 1: Expert
    expert_prompt = ROLE_EXPERT + "\n\n" + q_block
    r1 = call_ollama(model, expert_prompt, seed, num_ctx, timeout_s)
    expert_resp = r1["response"] if r1 else ""
    overflow_1 = r1 and (r1.get("truncated") or (r1.get("prompt_eval_count", 0)
                                                 + r1.get("eval_count", 0)
                                                 >= num_ctx - 10))

    # Stage 2: Critic
    critic_prompt = (ROLE_CRITIC + "\n\nThe Expert answered:\n"
                     + expert_resp + "\n\nOriginal question:\n" + q_block)
    r2 = call_ollama(model, critic_prompt, seed, num_ctx, timeout_s)
    critic_resp = r2["response"] if r2 else ""
    overflow_2 = r2 and (r2.get("truncated") or (r2.get("prompt_eval_count", 0)
                                                 + r2.get("eval_count", 0)
                                                 >= num_ctx - 10))

    # Stage 3: Evaluator
    eval_prompt = (ROLE_EVALUATOR + "\n\nOriginal question:\n" + q_block
                   + "\n\nExpert:\n" + expert_resp
                   + "\n\nCritic:\n" + critic_resp)
    r3 = call_ollama(model, eval_prompt, seed, num_ctx, timeout_s)
    eval_resp = r3["response"] if r3 else ""
    overflow_3 = r3 and (r3.get("truncated") or (r3.get("prompt_eval_count", 0)
                                                 + r3.get("eval_count", 0)
                                                 >= num_ctx - 10))

    # Stage 4: Synthesizer
    synth_prompt = (ROLE_SYNTHESIZER + "\n\nExpert:\n" + expert_resp
                    + "\n\nCritic:\n" + critic_resp
                    + "\n\nEvaluator:\n" + eval_resp
                    + "\n\nOriginal question:\n" + q_block)
    r4 = call_ollama(model, synth_prompt, seed, num_ctx, timeout_s)
    synth_resp = r4["response"] if r4 else ""
    overflow_4 = r4 and (r4.get("truncated") or (r4.get("prompt_eval_count", 0)
                                                 + r4.get("eval_count", 0)
                                                 >= num_ctx - 10))

    # Stage 5: Generalist (final answer letter)
    gen_prompt = (ROLE_GENERALIST + "\n\nFull deliberation summary:\n"
                  + synth_resp + "\n\nOriginal question:\n" + q_block)
    r5 = call_ollama(model, gen_prompt, seed, num_ctx, timeout_s)
    gen_resp = r5["response"] if r5 else ""
    overflow_5 = r5 and (r5.get("truncated") or (r5.get("prompt_eval_count", 0)
                                                 + r5.get("eval_count", 0)
                                                 >= num_ctx - 10))

    final_letter = parse_letter_from_json(gen_resp) if gen_resp else None
    correct_letter = row.get("correct_answer", "").strip().upper()

    stage_failures = sum(1 for resp in [r1, r2, r3, r4, r5] if resp is None)
    overflow = any([overflow_1, overflow_2, overflow_3, overflow_4, overflow_5])

    return {
        "question_id":        row["question_id"],
        "final_letter":       final_letter or "",
        "correct_letter":     correct_letter,
        "correct":            1 if (final_letter == correct_letter) else 0,
        "stage_failures":     stage_failures,
        "context_overflow":   int(bool(overflow)),
        "expert_tokens":      (r1 or {}).get("eval_count", 0),
        "critic_tokens":      (r2 or {}).get("eval_count", 0),
        "evaluator_tokens":   (r3 or {}).get("eval_count", 0),
        "synthesizer_tokens": (r4 or {}).get("eval_count", 0),
        "generalist_tokens":  (r5 or {}).get("eval_count", 0),
        "total_prompt_tokens_stage5": (r5 or {}).get("prompt_eval_count", 0),
        "transcript": {
            "expert":       expert_resp,
            "critic":       critic_resp,
            "evaluator":    eval_resp,
            "synthesizer":  synth_resp,
            "generalist":   gen_resp,
        },
    }


def run_one_model(model: str, questions: list[dict], seed: int, num_ctx: int,
                  timeout_s: int, output_dir: Path) -> None:
    slug = model.replace(":", "_").replace("/", "_")
    out_csv = output_dir / f"{slug}.csv"
    transcripts_dir = output_dir / f"{slug}__transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    t_model = time.time()
    for i, q in enumerate(questions):
        t0 = time.time()
        result = run_dpp_one_question(model, q, seed, num_ctx, timeout_s)
        elapsed = time.time() - t0

        # Save full transcript
        with (transcripts_dir / f"{q['question_id']}.json").open("w") as f:
            json.dump({"question": q, **result}, f, indent=2)
        transcript = result.pop("transcript", None)
        row = {**result, "elapsed_s": round(elapsed, 1)}
        rows.append(row)
        print(f"    [{i + 1}/{len(questions)}] qid={q['question_id']} "
              f"final={result['final_letter']} correct={result['correct']} "
              f"overflow={result['context_overflow']} "
              f"stage_fails={result['stage_failures']} "
              f"({elapsed:.1f}s)")

    with out_csv.open("w", newline="") as f:
        fields = [k for k in rows[0].keys()]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    total = time.time() - t_model
    print(f"  Saved: {out_csv}  ({total / 60:.1f} min, "
          f"{sum(r['correct'] for r in rows)}/{len(rows)} correct)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gpqa-jsonl", type=Path, required=True)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("./results/task10_dpp/"))
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout-s", type=int, default=300)
    args = ap.parse_args()

    with args.gpqa_jsonl.open() as f:
        questions = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions)} GPQA Diamond questions")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"Running {len(models)} models with num_ctx={args.num_ctx}")

    for model in models:
        print(f"\n=== {model} ===")
        run_one_model(model, questions, args.seed, args.num_ctx,
                      args.timeout_s, args.output_dir)

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
