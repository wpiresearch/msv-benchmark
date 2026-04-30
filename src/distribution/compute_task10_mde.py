#!/usr/bin/env python3
"""
compute_task10_mde.py
=====================

Computes the minimum detectable effect (MDE) per model for the Task 10
matched-lift analysis on the 9-model institutional cohort. The MDE
quantifies what per-model lift the 80-item panel WOULD have been able to
detect at conventional significance levels, given the empirical paired-
diff variability for each model.

Rationale
---------
Section 6.2 limitation 1 notes "the 80-question item count limits
statistical power; we address this via bootstrap CIs and subsampling
stability." Limitation 11 reports the matched-lift result: cohort mean
+0.021, range [-0.078, +0.125], 0/9 paired-bootstrap 95% CIs exclude
zero. A reviewer sympathetic to the paper might still ask: "what lift
WOULD you have detected at this n?" The MDE answers that.

This is not a pre-registered power analysis (paper is observational, not
a designed experiment). It is a post-hoc detectability calculation, of
the type that strengthens negative results by showing what magnitude of
effect was within the panel's reach. Reported under "limitations"
appendix or methodological notes; not a primary result.

Method
------
For each model, the paired-diff vector d_i = DPP_correct_i - FA_correct_i
on the FA-parsed subset has mean d_bar and SD s. Standard error of the
mean: SE = s / sqrt(n_matched). Two-sided MDE at alpha=0.05 is
1.96 * SE. One-sided MDE at alpha=0.05 is 1.645 * SE. For alpha=0.10
two-sided, use 1.645.

These MDE values translate directly to "the 95% CI half-width if d_bar
were zero." A model whose observed |d_bar| < MDE has insufficient
evidence to reject the null at that alpha.

Inputs
------
  --dpp-dir            Per-model Task 10 DPP CSVs (matches Task 10 analysis)
  --fa-dir             Per-model Forced-Answer Phase 1 CSVs
  --output-csv         Output CSV path

Outputs
-------
  Per-model rows with: model, n_matched, observed_lift, observed_sd,
  observed_se, mde_two_sided_05, mde_two_sided_10, mde_one_sided_05,
  cohort-aggregate row.

Worked example (qwen2.5:7b, n_matched=77):
  observed lift = -0.013, SD = 0.679 (paired-diff vector with values in
  {-1, 0, +1}), SE = 0.077, MDE_two-sided_05 = 0.151. The panel could
  have detected a lift of magnitude > 0.15 at alpha = 0.05; the observed
  lift (0.013) is far below this threshold.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


Z_TWO_SIDED_05 = 1.959964    # qnorm(0.975)
Z_TWO_SIDED_10 = 1.644854    # qnorm(0.95)
Z_ONE_SIDED_05 = 1.644854    # qnorm(0.95)


def normalize_model_name(filename: str) -> str:
    return Path(filename).stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dpp-dir", type=Path, required=True)
    ap.add_argument("--fa-dir",  type=Path, required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    args = ap.parse_args()

    dpp_csvs = sorted(args.dpp_dir.glob("*.csv"))
    if not dpp_csvs:
        print(f"ERROR: no DPP CSVs at {args.dpp_dir}")
        return 1

    rows = []
    all_diffs_pool: list[np.ndarray] = []

    print(f">>> Computing MDE for {len(dpp_csvs)} models")
    print()

    for dpp_csv in dpp_csvs:
        model = normalize_model_name(dpp_csv.name)
        fa_csv = args.fa_dir / dpp_csv.name
        if not fa_csv.exists():
            print(f"  [{model}] no FA CSV; skipping")
            continue

        dpp = pd.read_csv(dpp_csv)
        fa = pd.read_csv(fa_csv)
        for c in ("correct",):
            dpp[c] = dpp[c].astype(int)
            fa[c] = fa[c].astype(int)
        fa["parse_failure"] = fa["parse_failure"].astype(int)
        fa_parsed = fa[fa["parse_failure"] == 0]
        merged = pd.merge(
            dpp[["question_id", "correct"]].rename(columns={"correct": "dpp"}),
            fa_parsed[["question_id", "correct"]].rename(columns={"correct": "fa"}),
            on="question_id", how="inner",
        )
        diffs = (merged["dpp"] - merged["fa"]).to_numpy(dtype=float)
        n = len(diffs)
        if n < 2:
            print(f"  [{model}] n_matched={n}; skipping")
            continue
        observed = float(diffs.mean())
        sd = float(diffs.std(ddof=1))
        se = sd / np.sqrt(n)
        mde_05_two = Z_TWO_SIDED_05 * se
        mde_10_two = Z_TWO_SIDED_10 * se
        mde_05_one = Z_ONE_SIDED_05 * se

        all_diffs_pool.append(diffs)
        rows.append({
            "model":               model,
            "n_matched":           n,
            "observed_lift":       round(observed, 4),
            "observed_sd":         round(sd, 4),
            "observed_se":         round(se, 4),
            "mde_two_sided_05":    round(mde_05_two, 4),
            "mde_two_sided_10":    round(mde_10_two, 4),
            "mde_one_sided_05":    round(mde_05_one, 4),
            "observed_below_mde_05_twosided": int(abs(observed) < mde_05_two),
        })
        print(f"  [{model:>20s}] n={n:>3d}  d_bar={observed:+.4f}  "
              f"sd={sd:.4f}  se={se:.4f}  mde_05_two={mde_05_two:+.4f}  "
              f"observed_below_mde={'yes' if abs(observed) < mde_05_two else 'NO'}")

    # Cohort aggregate
    if all_diffs_pool:
        pooled = np.concatenate(all_diffs_pool)
        n = len(pooled)
        observed = float(pooled.mean())
        sd = float(pooled.std(ddof=1))
        se = sd / np.sqrt(n)
        rows.append({
            "model":               "COHORT_POOLED",
            "n_matched":           n,
            "observed_lift":       round(observed, 4),
            "observed_sd":         round(sd, 4),
            "observed_se":         round(se, 4),
            "mde_two_sided_05":    round(Z_TWO_SIDED_05 * se, 4),
            "mde_two_sided_10":    round(Z_TWO_SIDED_10 * se, 4),
            "mde_one_sided_05":    round(Z_ONE_SIDED_05 * se, 4),
            "observed_below_mde_05_twosided": int(abs(observed) < Z_TWO_SIDED_05 * se),
        })
        print()
        print(f"  [COHORT_POOLED]      n={n:>3d}  d_bar={observed:+.4f}  "
              f"sd={sd:.4f}  se={se:.4f}  mde_05_two={Z_TWO_SIDED_05 * se:+.4f}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print()
    print(f">>> Wrote {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
