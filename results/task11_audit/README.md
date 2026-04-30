Contains the 8 canonical CSVs

**The 8 canonical CSVs:**

| File                                  |  Bytes | Purpose                                                      |
| ------------------------------------- | -----: | ------------------------------------------------------------ |
| `task11_metacognitive_efficiency.csv` |  2,788 | Per-model d-hat, Type-2 AUC, MC with bootstrap CIs (the headline output: contains the corrected Haiku Type-2 AUC = 0.747) |
| `convergence_matrix.csv`              |  2,126 | 23-model × 7-signal cross-task convergence matrix with z-scores (data underlying paper Figure 4) |
| `task1_delegation_curves.csv`         |  4,759 | Per-model delegation rate × difficulty bin                   |
| `task1_delegation_curves_slopes.csv`  |    979 | Per-model delegation slope (one row per model)               |
| `task2_coherence.csv`                 |  1,594 | Per-model declared-routing Spearman ρ for Task 2             |
| `verbosity_stats.csv`                 | 16,828 | Per-model × per-task output token statistics                 |
| `verbosity_vs_mc.csv`                 |    297 | Verbosity-vs-MC correlation summary                          |
| `run_catalog.csv`                     | 56,221 | The catalog of which (model, task) runs are present (provenance trail) |

The directory also contains 6 PNG figures (`convergence_matrix.png`, `d_hat_vs_type2auc.png`, `task2_coherence_scatter.png`, `verbosity_distribution.png`, `verbosity_vs_mc_scatter.png`, `delegation_by_difficulty.png`) — the first two are paper figures and should also be copied to `results/figures/` (per AUDIT_FINDINGS.md Phase 4); the others are diagnostic.

The 8 CSVs are what go into `results/task11_audit/` because they're the **paper-bound provenance for Appendix `app:task11_audit`** — every quantitative claim in that appendix can be traced to one of these CSVs. 
