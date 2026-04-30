# Rank Divergence Audit

## Motivation

This audit reconciles an inconsistency detected during reviewer response.
The main paper reported Kendall tau = 0.018 between the declarative ECE
ranking and the behavioral Delegation AUC-ROC ranking on the Kaggle
cohort, described as "indistinguishable from zero." A reviewer
subsequently requested bootstrap confidence intervals on that number.
The new bootstrap script produced a point estimate of tau ~ +0.29, not
+0.018, on what we initially believed was the same subset.

We audited the analysis and found that the original tau = 0.018 claim
depends on two specific analysis choices:

  1. **AUC target**: the original script used
     `deleg_auc_vs_hardness` (delegation AUC against *dataset-level
     question difficulty*), NOT `deleg_auc_vs_own_err` (AUC against the
     model's own error probability).
  2. **Inclusion rule**: all 11 models with both metrics non-NaN were
     included, with no minimum-answered threshold.

These are both defensible choices, but together they operationalize a
different estimand than the paper's conceptual claim. The paper's text
describes the behavioral metric as measuring "whether the model's
routing behavior tracks its own failure probability" (Section 5.1), which
is the own-error AUC, not the hardness AUC.

## Sensitivity grid

Six-cell grid of point estimates under the combinations of (AUC target)
x (inclusion rule):

| AUC target | Inclusion rule | n | tau | p | rho |
|---|---|---|---|---|---|
| Hardness | all-computable (original paper result) | 11 | +0.018 | 1.000 | +0.018 |
| Hardness | n_answered >= 20, non-degenerate | 10 | -0.067 | 0.862 | -0.091 |
| Hardness | n_answered >= 5, non-degenerate | 11 | +0.018 | 1.000 | +0.018 |
| **Own-error** | **all-computable (proposed primary)** | **11** | **+0.200** | **0.445** | **+0.255** |
| Own-error | n_answered >= 20, non-degenerate | 10 | +0.289 | 0.291 | +0.382 |
| Own-error | n_answered >= 5, non-degenerate | 11 | +0.200 | 0.445 | +0.255 |

## Decision

Under the conceptual claim that the paper is testing --- whether
answered-conditional ECE and delegation-based error awareness yield
different model rankings --- **own-error AUC is the correct
operationalization**. The revised primary estimand is:

  > tau = +0.20 (n = 11, all models with computable own-error AUC and ECE)

with a sensitivity analysis showing tau ranges from -0.07 to +0.29
across reasonable choices of AUC target and inclusion rule. The
hardness-AUC tau = +0.018 is demoted to a sensitivity analysis row
under a different estimand.

## Downstream paper changes

1. **Section 5.1 and the comparative Section 5.2**: the behavioral AUC
   should be explicitly labeled as "Delegation AUC-ROC against own
   error" rather than "Delegation AUC-ROC", and the distinction from
   hardness AUC should be made explicit.
2. **Section 5.5 and Section 6.1**: the tau = 0.018 phrasing
   ("indistinguishable from zero") is no longer correct as a primary
   finding. Revise to "tau = 0.20 (95% CI pending bootstrap), moderately
   positive but with substantial rank reversals" and cite named
   examples.
3. **Conclusion**: the "different rankings" claim remains, but the
   quantitative number changes from 0.018 to 0.20 with the own-error
   framing.
4. **Appendix G (Reproducibility)**: add a pointer to
   `rank_divergence_audit.csv` and the full sensitivity grid.
5. **Appendix D (Kaggle cohort)**: add the sensitivity table above.

The rank-divergence figure (`rank_reversal_scatter`) was already built
using own-error AUC (from `delegate_game_metrics.csv` column
`deleg_auc_vs_hardness` merged against a separate calculation that used
own-error — confirmed in a subsequent independent regeneration), so the
figure itself is valid; the caption will need its tau annotation updated.
