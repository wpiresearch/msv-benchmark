#!/usr/bin/env python3
"""
test_ece_consistency.py
=======================

Verifies that all four scripts in the bundle that compute ECE produce
IDENTICAL output on a fixed test panel. This test exists because the
ECE estimator is implemented inline in each script (not via a shared
utility module), and we want a machine-checkable guarantee that the
four implementations have not drifted apart.

DESIGN RATIONALE
----------------

Why inline implementations rather than a shared utility module?

This bundle is a reproducibility artifact, not a maintained library.
Reviewers will download a zip, possibly run only one script, and audit
each file in isolation. A shared utility module would force them to
chase imports across files to understand what ECE means, and would add
a deployment surface (PYTHONPATH, file-location coupling) that doesn't
exist with self-contained scripts.

The legitimate concern with inline duplication ("one script silently
keeps an older estimator") is addressed by THIS test rather than by
preventing duplication structurally. Run this test after any change
to any ECE function in the bundle. If it fails, the four implementations
have drifted and one or more must be brought back into agreement.

WHAT IS LABEL-GROUPED ECE?
--------------------------

Per paper Section 4.3:

    "Expected Calibration Error with 4 bins, one per discrete confidence
    level. Since models report confidence on a 1-4 scale, using more
    bins than confidence levels produces degenerate empty bins; we
    therefore bin by confidence level directly. Confidence ratings 1-4
    are mapped linearly to probabilities {0.25, 0.50, 0.75, 1.00}."

Formally:

    ECE = sum_{k in {1,2,3,4}} (n_k / n) * |acc_k - p_k|

where:
    k       = declared confidence label
    p_k     = k / 4   (paper-canonical four-choice mapping)
    acc_k   = empirical accuracy among trials with confidence label k
    n_k     = number of trials with confidence label k
    n       = total number of valid trials

This is NOT equal-width binning on [0, 1]. The previous (incorrect)
implementation used four equal-width bins and would silently merge
confidence levels 3 and 4 into the bin [0.75, 1.00] under the canonical
mapping, producing different ECE values whenever within-bin accuracy
varies between merged labels (the typical case for well-calibrated
models, where confidence 4 should be more accurate than confidence 3).

USAGE
-----

    python test_ece_consistency.py

Exit code 0 if all four implementations produce identical output on
all test cases. Exit code 1 if any implementation diverges.

REQUIREMENTS
------------

The four scripts must be in the same directory as this test, or the
SCRIPTS_DIR variable below must be adjusted to point at them.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np


# Adjust SCRIPTS_DIR if scripts live elsewhere relative to this test.
# Default: scripts are siblings of this test file (i.e., in the same dir).
SCRIPTS_DIR = Path(__file__).parent

ECE_FUNCTION_LOCATIONS = [
    # (script filename, module-level function name, input convention)
    # Input convention is one of:
    #   "raw":  function takes (correct, confidence) where confidence is
    #           integer-valued labels in {1, 2, 3, 4}
    #   "prob": function takes (..., confidence_probs) where confidence
    #           has been pre-normalized to {0.25, 0.50, 0.75, 1.00}
    ("compute_bootstrap_ci.py",         "compute_ece",  "raw"),
    ("compute_rank_divergence_ci.py",   "compute_ece",  "raw"),
    ("analyze_kaggle_cohort.py",        "_ece",         "prob"),
    ("compute_comparative_baselines.py", "compute_ece", "prob"),
]


def load_function(script_filename, fn_name):
    """Import the named function from the named script."""
    script_path = SCRIPTS_DIR / script_filename
    if not script_path.exists():
        raise FileNotFoundError(f"Cannot find {script_path}")
    spec = importlib.util.spec_from_file_location(
        script_filename.replace(".py", ""), script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, fn_name):
        raise AttributeError(
            f"{script_filename} has no function named {fn_name}"
        )
    return getattr(module, fn_name)


def label_grouped_ece_reference(correct, confidence_labels):
    """The reference implementation, written inline for clarity.

    This is the answer the four bundle implementations should agree on.
    """
    correct = np.asarray(correct, dtype=float)
    confidence_labels = np.asarray(confidence_labels, dtype=int)
    valid = ~np.isnan(correct.astype(float))
    correct = correct[valid]
    labels = confidence_labels[valid]
    n = len(correct)
    if n < 2:
        return float("nan")
    ece = 0.0
    for k in (1, 2, 3, 4):
        mask = (labels == k)
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_k = k / 4.0
        acc_k = correct[mask].mean()
        ece += (nb / n) * abs(acc_k - p_k)
    return ece


# ============================================================================
# Test cases
# ============================================================================
#
# Each test case is a tuple of (name, correct, confidence_labels, expected_ece).
# The expected_ece is computed independently from the formula in the docstring,
# not by running any of the bundle implementations. This is the ground truth.
#
# Test cases are chosen to exercise specific failure modes:
#   1. equal_acc_across_levels: a coincidence case where label-grouped and
#      old equal-width ECE happen to agree (constant within-bin accuracy).
#      Catches catastrophic regressions but not the binning bug specifically.
#   2. differentiating_3_vs_4: conf 3 acc=1.0, conf 4 acc=0.0. The ECE
#      values diverge dramatically (0.625 vs 0.375) between label-grouped
#      and equal-width-with-merged-last-bin.
#   3. all_four_levels_unequal: each confidence level has different
#      accuracy. Tests that no two labels are silently merged.
#   4. only_one_level: edge case with single confidence label populated.
#   5. perfect_calibration: per-label accuracy matches per-label probability
#      exactly, ECE should be 0.

TEST_CASES = [
    (
        "equal_acc_across_levels",
        [0, 1, 0, 1, 1, 0],
        [3, 3, 4, 4, 1, 2],
        # Label-grouped:
        # k=1: n=1, acc=1.0, p=0.25, contrib=(1/6)*0.75=0.125
        # k=2: n=1, acc=0.0, p=0.50, contrib=(1/6)*0.50=0.0833
        # k=3: n=2, acc=0.5, p=0.75, contrib=(2/6)*0.25=0.0833
        # k=4: n=2, acc=0.5, p=1.00, contrib=(2/6)*0.50=0.1667
        # Total=0.4583
        0.45833333333333333,
    ),
    (
        "differentiating_3_vs_4",
        [1, 1, 0, 0],
        [3, 3, 4, 4],
        # Label-grouped:
        # k=3: n=2, acc=1.0, p=0.75, contrib=(2/4)*0.25=0.125
        # k=4: n=2, acc=0.0, p=1.00, contrib=(2/4)*1.00=0.500
        # Total=0.625
        # (Old equal-width-bug would give 0.375 here. Critical test.)
        0.625,
    ),
    (
        "all_four_levels_unequal",
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1],
        [1, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        # Label-grouped:
        # k=1: n=2, acc=0.0, p=0.25, contrib=(2/11)*0.25 = 0.0455
        # k=2: n=2, acc=0.5, p=0.50, contrib=(2/11)*0.0  = 0.0
        # k=3: n=3, acc=2/3, p=0.75, contrib=(3/11)*0.0833 = 0.0227
        # k=4: n=4, acc=1.0, p=1.00, contrib=(4/11)*0.0  = 0.0
        # Total ~ 0.0682
        (2/11)*0.25 + (2/11)*0.0 + (3/11)*abs(2/3 - 0.75) + (4/11)*0.0,
    ),
    (
        "only_one_level",
        [1, 1, 0, 1, 0],
        [3, 3, 3, 3, 3],
        # k=3: n=5, acc=0.6, p=0.75, contrib=(5/5)*0.15=0.15
        0.15,
    ),
    (
        "perfect_calibration",
        # 4 trials at conf 1 (25% acc), 4 at conf 2 (50%), 4 at conf 3 (75%), 4 at conf 4 (100%)
        [0, 0, 0, 1] + [0, 0, 1, 1] + [0, 1, 1, 1] + [1, 1, 1, 1],
        [1, 1, 1, 1] + [2, 2, 2, 2] + [3, 3, 3, 3] + [4, 4, 4, 4],
        # All per-label contributions are zero
        0.0,
    ),
]


def run_tests():
    print("Loading ECE functions from each bundle script...")
    fns = {}
    for script, fn_name, convention in ECE_FUNCTION_LOCATIONS:
        try:
            fn = load_function(script, fn_name)
            fns[script] = (fn, convention)
            print(f"  loaded: {script}::{fn_name}  (input convention: {convention})")
        except Exception as e:
            print(f"  ERROR loading {script}::{fn_name}: {e}")
            return False
    print()

    all_passed = True
    for case_name, correct, conf_labels, expected_ece in TEST_CASES:
        print(f"Test case: {case_name}")
        print(f"  correct = {correct}")
        print(f"  conf    = {conf_labels}")
        print(f"  expected ECE (label-grouped reference): {expected_ece:.10f}")

        # Independent reference computation
        ref = label_grouped_ece_reference(correct, conf_labels)
        if abs(ref - expected_ece) > 1e-9:
            print(f"  REFERENCE MISMATCH: ref={ref:.10f}, expected={expected_ece:.10f}")
            print("  (This indicates a bug in this test file's expected values, not in the bundle scripts.)")
            all_passed = False
            continue

        correct_arr = np.asarray(correct, dtype=float)
        conf_int = np.asarray(conf_labels, dtype=float)  # raw labels as floats
        conf_prob = np.asarray(conf_labels, dtype=float) / 4.0  # already-mapped

        case_passed = True
        for script, (fn, convention) in fns.items():
            if convention == "raw":
                result = fn(correct_arr, conf_int)
            elif convention == "prob":
                # analyze_kaggle_cohort.py and compute_comparative_baselines.py
                # signatures differ in argument order
                if script == "analyze_kaggle_cohort.py":
                    result = fn(conf_prob, correct_arr)
                else:
                    result = fn(correct_arr, conf_prob)
            else:
                raise ValueError(f"unknown convention: {convention}")

            ok = abs(result - expected_ece) < 1e-9
            mark = "OK" if ok else "FAIL"
            print(f"    {script:38s} -> {result:.10f}  [{mark}]")
            if not ok:
                case_passed = False

        all_passed = all_passed and case_passed
        print()

    if all_passed:
        print("=" * 70)
        print("ALL TESTS PASSED")
        print("All four ECE implementations agree with the label-grouped reference")
        print("on every test case.")
        print("=" * 70)
        return True
    else:
        print("=" * 70)
        print("AT LEAST ONE TEST FAILED")
        print("Some implementation has drifted. Audit the failing scripts.")
        print("=" * 70)
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
