#!/usr/bin/env python3
"""
extract_kaggle_outputs.py
=========================

Extracts per-task, per-model results from the Kaggle Benchmarks SDK run
archive (`outputs_logs/`) into an analysis-ready directory structure.

Input layout (`--input-dir`):
    <input-dir>/
        t<NN>-<task_slug>-<provider>_<model>.zip   (contains results.csv + task.json + run.json)
        t<NN>-<task_slug>-<provider>_<model>.log

Output layout (`--output-dir`):
    <output-dir>/
        per_task/
            t01_delegate_game.csv        (all models x all questions, long form)
            t02_declared_probe.csv
            ...
            t11_mc_binary_pairs.csv
        per_model/
            claude-haiku-4-5-20251001/
                t01_delegate_game.csv    (this model's 80 Task 1 rows)
                t02_declared_probe.csv
                ...
        run_metadata.csv                 (one row per task-model pair: completion %,
                                          parse failures, mean score, budget failure flag,
                                          canonical model name, timestamp)
        leaderboard_reconciled.csv       (platform leaderboard scores vs re-derived from csvs)
        extraction_log.txt               (any warnings/errors)

Usage:
    python extract_kaggle_outputs.py \
        --input-dir ./outputs_logs \
        --output-dir ./kaggle_extracted \
        --leaderboard ./leaderboard.csv

"""

import argparse
import csv
import io
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path


# --- Canonical model name mapping (zip-filename -> leaderboard name) ---
MODEL_NAME_MAP = {
    "anthropic_claude-haiku-4-520251001":    "claude-haiku-4-5-20251001",
    "anthropic_claude-opus-4-120250805":     "claude-opus-4-1-20250805",
    "anthropic_claude-opus-4-6default":      "claude-opus-4-6-default",
    "deepseek-ai_deepseek-r1-0528":          "deepseek-r1-0528",
    "deepseek-ai_deepseek-v3.2":             "deepseek-v3.2",
    "google_gemini-2.0-flash-lite":          "gemini-2.0-flash-lite-001",
    "google_gemini-2.5-flash":               "gemini-2.5-flash",
    "google_gemini-3.1-pro-preview":         "gemini-3.1-pro-preview",
    "google_gemma-3-12b":                    "gemma-3-12b-it",
    "google_gemma-3-1b":                     "gemma-3-1b-it",
    "google_gemma-3-27b":                    "gemma-3-27b-it",
    "google_gemma-3-4b":                     "gemma-3-4b-it",
    "google_gemma-4-26b-a4b":                "gemma-4-26b-a4b-it",
    "google_gemma-4-31b":                    "gemma-4-31b-it",
    "openai_gpt-5.4-2026-03-05":             "gpt-5.4-2026-03-05",
    "openai_gpt-5.4-mini-2026-03-17":        "gpt-5.4-mini-2026-03-17",
    "openai_gpt-5.4-nano-2026-03-17":        "gpt-5.4-nano-2026-03-17",
    "openai_gpt-oss-20b":                    "gpt-oss-20b",
    "qwen_qwen3-235b-a22b-instruct-2507":    "qwen3-235b-a22b-instruct-2507",
    "qwen_qwen3-coder-480b-a35b-instruct":   "qwen3-coder-480b-a35b-instruct",
    "qwen_qwen3-next-80b-a3b-instruct":      "qwen3-next-80b-a3b-instruct",
    "qwen_qwen3-next-80b-a3b-thinking":      "qwen3-next-80b-a3b-thinking",
    "zai_glm-5":                             "glm-5",
}

# --- Task identifier -> short slug used in CSV filenames ---
TASK_SLUGS = {
    "t01": ("t01-msv_delegate_game",      "t01_delegate_game"),
    "t02": ("t02-msv_declared_probe",     "t02_declared_probe"),
    "t03": ("t03-msv_second_chance",      "t03_second_chance"),
    "t04": ("t04-msv_confidence_entropy", "t04_confidence_entropy"),
    "t05": ("t05-msv_teammate_delegate",  "t05_teammate_delegate"),
    "t06": ("t06-msv_behavioral_er",      "t06_behavioral_er"),
    "t07": ("t07-msv_behavioral_ci",      "t07_behavioral_ci"),
    "t08": ("t08-msv_behavioral_em",      "t08_behavioral_em"),
    "t09": ("t09-msv_behavioral_pi",      "t09_behavioral_pi"),
    "t10": ("t10-msv_dpp_sequence",       "t10_dpp_sequence"),
    "t11": ("t11-msv_mc_binary_pairs",    "t11_mc_binary_pairs"),
}

# List of known task slugs, used by the robust parse_stem below
_TASK_SLUG_LIST = [full for (full, _short) in TASK_SLUGS.values()]

# Explicit corrections for malformed filenames we have observed.
# Keys are the raw "provider_model" portion after the task prefix is stripped;
# values are the canonical leaderboard model name.
# These come from manual inspection of the Kaggle run archives.
FILENAME_ALIASES = {
    "qwen_qwen3-next-80b-a3b-instrut":            "qwen3-next-80b-a3b-instruct",  # t03: missing 'c' in 'instruct'
    "anthropic_claude-opus-4-120":                "claude-opus-4-1-20250805",     # t04: truncated date
    "deepseek-ai_deepseek-r1-052":                "deepseek-r1-0528",             # t04 & t05: truncated '052' -> '0528'
    "qwen_qwen3-235b-a22b-instrut":               "qwen3-235b-a22b-instruct-2507",# t04: missing 'c' + date
    "anthropic_claude-haiku-4-520":               "claude-haiku-4-5-20251001",    # t05: truncated date
    "qwen_qwen3-235b-a22b-instruct-2507.":        "qwen3-235b-a22b-instruct-2507",# t05: trailing dot
    "qwen_qwen3-coder-480b-a3":                   "qwen3-coder-480b-a35b-instruct", # t05: truncated
    "qwen_qwen3-next-80b-a3b-insstruct":          "qwen3-next-80b-a3b-instruct",  # t05: doubled 's'
    "anthropic_claude-haiku-4-52025100":          "claude-haiku-4-5-20251001",    # t07: missing trailing '1'
    "anthropic_claude-haiku-4-520251":            "claude-haiku-4-5-20251001",    # t08: truncated date
    "anthropic_claude-claude-haiku-4-520251001":  "claude-haiku-4-5-20251001",    # t11: doubled 'claude-'
}


def parse_stem(stem: str):
    """Return (task_id, task_slug_full, provider_model_raw) or None.

    Handles both '-' and '_' as the separator between task slug and model
    (t09 uses '_' for many of its filenames).
    """
    for task_id, (task_slug_full, _short) in TASK_SLUGS.items():
        for sep in ("-", "_"):
            prefix = f"{task_slug_full}{sep}"
            if stem.startswith(prefix):
                return task_id, task_slug_full, stem[len(prefix):]
    return None


def extract_run_info(zip_path: Path):
    """Open a Kaggle run zip and return (results_csv_text, task_json, run_json_meta).

    `run_json_meta` is a dict with top-level metadata only (state, startTime,
    endTime, result value, num requests) to keep memory low; we do not load the
    full conversation transcripts into Python memory.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if not n.startswith("__MACOSX/")]
        csv_name  = next((n for n in names if n.endswith("_results.csv")), None)
        task_name = next((n for n in names if n.endswith(".task.json")),   None)
        run_name  = next((n for n in names if n.endswith(".run.json")),    None)

        csv_text = zf.read(csv_name).decode("utf-8") if csv_name else None
        task_meta = json.loads(zf.read(task_name)) if task_name else None

        run_meta = None
        if run_name:
            run_full = json.loads(zf.read(run_name))
            convs = run_full.get("conversations", [])
            n_requests = sum(len(c.get("requests", [])) for c in convs)
            results = run_full.get("results", [])
            result_value = None
            if results and isinstance(results[0].get("numericResult"), dict):
                result_value = results[0]["numericResult"].get("value")
            run_meta = {
                "state":          run_full.get("state"),
                "startTime":      run_full.get("startTime"),
                "endTime":        run_full.get("endTime"),
                "num_requests":   n_requests,
                "result_value":   result_value,
                "pyRunId":        run_full.get("pyRunId"),
                "modelVersionSlug": convs[0].get("modelVersionSlug") if convs else None,
            }

    return csv_text, task_meta, run_meta


def parse_log_for_completion(log_path: Path):
    """Extract completion info from the .log file.

    Looks for lines like:
      "Mean score: 0.6712 | Delegate rate: 45.00% | Parse failures: 0/80"
      "Completion: 80/80 (100%)"
      "[prompt failure] PermissionDeniedError: Error code: 403 - {...budget...}"
    Returns a dict with mean_score, completion_count, completion_total,
    parse_failures, budget_failure (bool), other_failure (str or None).
    """
    info = {
        "mean_score": None,
        "completion_count": None,
        "completion_total": None,
        "parse_failures": None,
        "budget_failure": False,
        "other_failure": None,
    }
    if not log_path.exists():
        info["other_failure"] = "log file missing"
        return info

    text = log_path.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"Mean score:\s*([0-9.]+)", text)
    if m:
        info["mean_score"] = float(m.group(1))
    m = re.search(r"Parse failures:\s*(\d+)/(\d+)", text)
    if m:
        info["parse_failures"] = int(m.group(1))
    m = re.search(r"Completion:\s*(\d+)/(\d+)", text)
    if m:
        info["completion_count"] = int(m.group(1))
        info["completion_total"] = int(m.group(2))

    # Budget/permission failure detection
    if ("PermissionDeniedError" in text or "exceeds your available quota" in text
            or "BudgetExceeded" in text):
        info["budget_failure"] = True
    # Context window detection
    if "context_length_exceeded" in text or "context window" in text.lower():
        info["other_failure"] = "context_length"
    # Generic prompt failure (catches rate-limit and other cases)
    if "[prompt failure]" in text and not info["budget_failure"]:
        if info["other_failure"] is None:
            # First 200 chars of the failure line for debugging
            m = re.search(r"\[prompt failure\][^\n]*", text)
            if m:
                info["other_failure"] = m.group(0)[:200]

    return info


def read_leaderboard(path: Path):
    """Return a dict: (model, task_slug_full) -> numerical_result."""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            model = row.get("Model", "").strip()
            task  = (row.get("Task_Name") or "").strip()
            val   = row.get("Numerical_Result", "").strip()
            if not model or not val:
                continue
            try:
                val = float(val)
            except ValueError:
                continue
            out[(model, task)] = val
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir",   required=True, type=Path, help="Directory containing the .zip + .log run archives")
    ap.add_argument("--output-dir",  required=True, type=Path, help="Directory to write extracted results")
    ap.add_argument("--leaderboard", type=Path, default=None, help="Optional path to leaderboard CSV for reconciliation")
    args = ap.parse_args()

    in_dir  = args.input_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_task").mkdir(exist_ok=True)
    (out_dir / "per_model").mkdir(exist_ok=True)

    log_f = open(out_dir / "extraction_log.txt", "w")
    def log(msg, echo=True):
        log_f.write(msg + "\n")
        if echo:
            print(msg)

    log(f"Input dir : {in_dir}")
    log(f"Output dir: {out_dir}")

    # Inventory run zips
    zips = sorted(in_dir.glob("t*.zip"))
    log(f"Found {len(zips)} run zips")

    run_rows = []                      # one per task-model pair
    task_accumulator = defaultdict(list)   # task_slug_full -> list of (canonical_model, rows_list, header)
    task_headers = {}                  # task_slug_full -> header columns (first seen wins)

    for zf_path in zips:
        stem = zf_path.stem
        parsed = parse_stem(stem)
        if parsed is None:
            log(f"  [WARN] skipping unparseable filename: {stem}")
            continue
        task_id, task_slug_full, provider_model_raw = parsed

        if task_id not in TASK_SLUGS:
            log(f"  [WARN] unknown task id: {task_id} in {stem}")
            continue
        task_slug_short = TASK_SLUGS[task_id][1]

        canonical = MODEL_NAME_MAP.get(provider_model_raw) or FILENAME_ALIASES.get(provider_model_raw)
        if canonical is None:
            log(f"  [WARN] unmapped model: {provider_model_raw} in {stem}")
            continue
        if provider_model_raw not in MODEL_NAME_MAP:
            log(f"  [INFO] applied alias: '{provider_model_raw}' -> '{canonical}' (file: {stem})")

        # Extract run archive
        csv_text, task_meta, run_meta = extract_run_info(zf_path)

        # Parse companion log (may be missing for a few runs)
        log_path = in_dir / (stem + ".log")
        log_info = parse_log_for_completion(log_path)

        # Derive row count from CSV (authoritative for "how many questions did we score?")
        if csv_text:
            reader = csv.reader(io.StringIO(csv_text))
            header = next(reader, None)
            rows   = [r for r in reader if any(c.strip() for c in r)]
        else:
            header, rows = None, []

        # Accumulate per-task (long-form) - prepend canonical model col
        if header is not None:
            if task_slug_full not in task_headers:
                task_headers[task_slug_full] = header
            else:
                # Sanity: headers should match across models for the same task
                if task_headers[task_slug_full] != header:
                    log(f"  [WARN] header mismatch for {task_slug_full} model={canonical}; "
                        f"keeping original header, padding rows if needed")
            task_accumulator[task_slug_full].append((canonical, header, rows))

        # Write per-model CSV
        model_dir = out_dir / "per_model" / canonical
        model_dir.mkdir(parents=True, exist_ok=True)
        if csv_text is not None:
            (model_dir / f"{task_slug_short}.csv").write_text(csv_text, encoding="utf-8")

        # Metadata row
        run_rows.append({
            "task_id":           task_id,
            "task_slug":         task_slug_full,
            "model":             canonical,
            "provider_model_raw": provider_model_raw,
            "n_rows_csv":        len(rows),
            "log_mean_score":    log_info["mean_score"],
            "log_completion_count": log_info["completion_count"],
            "log_completion_total": log_info["completion_total"],
            "log_parse_failures":   log_info["parse_failures"],
            "budget_failure":    log_info["budget_failure"],
            "other_failure":     log_info["other_failure"],
            "run_state":         (run_meta or {}).get("state"),
            "run_start":         (run_meta or {}).get("startTime"),
            "run_end":           (run_meta or {}).get("endTime"),
            "run_num_requests":  (run_meta or {}).get("num_requests"),
            "run_result_value":  (run_meta or {}).get("result_value"),
            "task_version":      (task_meta or {}).get("versionNumber"),
        })

    # Build reverse map: task_slug_full -> task_slug_short (cheaper than re-iterating)
    FULL_TO_SHORT = {full: short for (full, short) in TASK_SLUGS.values()}

    # --- Write long-form per-task CSVs ---
    for task_slug_full, entries in task_accumulator.items():
        task_slug_short = FULL_TO_SHORT[task_slug_full]
        out_path = out_dir / "per_task" / f"{task_slug_short}.csv"
        reference_header = task_headers[task_slug_full]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model"] + reference_header)
            for canonical_model, header, rows in sorted(entries, key=lambda x: x[0]):
                # Align rows to reference header if they differ (defensive)
                if header == reference_header:
                    for r in rows:
                        w.writerow([canonical_model] + r)
                else:
                    # Column-by-column remap
                    idx_map = {c: (header.index(c) if c in header else None) for c in reference_header}
                    for r in rows:
                        aligned = [r[idx_map[c]] if idx_map[c] is not None else "" for c in reference_header]
                        w.writerow([canonical_model] + aligned)
        log(f"  wrote per_task/{task_slug_short}.csv ({sum(len(e[2]) for e in entries)} rows across {len(entries)} models)")

    # --- Write run metadata CSV ---
    run_rows.sort(key=lambda r: (r["task_id"], r["model"]))
    with open(out_dir / "run_metadata.csv", "w", newline="", encoding="utf-8") as f:
        if run_rows:
            w = csv.DictWriter(f, fieldnames=list(run_rows[0].keys()))
            w.writeheader()
            w.writerows(run_rows)
    log(f"\nWrote run_metadata.csv with {len(run_rows)} rows (task × model pairs)")

    # --- Leaderboard reconciliation (if provided) ---
    #
    # Two different scores appear in the data for a task run:
    #   log_mean_score   = mean over *completed* trials (from the log file's "Mean score: X")
    #   run_result_value = mean over *all* scheduled trials, missing ones counted as 0
    #                      (from run.json's results[0].numericResult.value)
    # The Kaggle leaderboard score matches run_result_value, not log_mean_score.
    # When a run is fully complete these are equal; when a run is partial (budget
    # failure or other prompt failure), they diverge. We reconcile against
    # run_result_value because that is what the platform publishes.
    if args.leaderboard and args.leaderboard.exists():
        lb = read_leaderboard(args.leaderboard)
        log(f"\nLeaderboard loaded: {len(lb)} entries")
        rec_rows = []
        for r in run_rows:
            task_slug = r["task_slug"]
            model     = r["model"]
            lb_score  = lb.get((model, task_slug))
            platform_score     = r["run_result_value"]    # the official Kaggle number
            completed_mean     = r["log_mean_score"]       # mean over completed trials only
            diff_platform      = None if (lb_score is None or platform_score is None) else round(lb_score - platform_score, 4)
            rec_rows.append({
                "task_slug":            task_slug,
                "model":                model,
                "leaderboard_score":    lb_score,
                "platform_score":       platform_score,
                "completed_mean":       completed_mean,
                "diff_vs_platform":     diff_platform,
                "n_rows_csv":           r["n_rows_csv"],
                "completion":           f"{r['log_completion_count']}/{r['log_completion_total']}"
                                        if r["log_completion_count"] is not None else "",
                "budget_failure":       r["budget_failure"],
                "other_failure":        r["other_failure"],
            })
        with open(out_dir / "leaderboard_reconciled.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rec_rows[0].keys()))
            w.writeheader()
            w.writerows(rec_rows)
        mismatches = [r for r in rec_rows if r["diff_vs_platform"] is not None and abs(r["diff_vs_platform"]) > 1e-3]
        log(f"Leaderboard vs platform-score reconciliation: {len(mismatches)} rows with |diff| > 1e-3")
        for m in mismatches[:20]:
            log(f"  {m['task_slug']:<32} {m['model']:<35} "
                f"lb={m['leaderboard_score']} platform={m['platform_score']} diff={m['diff_vs_platform']}")
        if len(mismatches) > 20:
            log(f"  ... {len(mismatches) - 20} more (see leaderboard_reconciled.csv)")

    # --- Print summary ---
    log(f"\n=== SUMMARY ===")
    by_task = defaultdict(int)
    for r in run_rows:
        by_task[r["task_slug"]] += 1
    for task_slug in sorted(by_task.keys()):
        log(f"  {task_slug}: {by_task[task_slug]} model runs")

    budget_failed = [r for r in run_rows if r["budget_failure"]]
    log(f"\nBudget failures: {len(budget_failed)}")
    for r in budget_failed:
        log(f"  {r['task_slug']:<32} {r['model']:<35} "
            f"completed {r['log_completion_count']}/{r['log_completion_total']}")

    other_failed = [r for r in run_rows if r["other_failure"]]
    log(f"\nOther failures: {len(other_failed)}")
    for r in other_failed:
        log(f"  {r['task_slug']:<32} {r['model']:<35} {r['other_failure'][:80]}")

    log_f.close()


if __name__ == "__main__":
    main()
