### 3. `verbosity_distribution.png` — DO NOT RECOMMEND

A horizontal bar chart of mean output tokens per response across the 23 models, showing the verbose-CoT cluster (DeepSeek R1, Qwen3-Thinking, GLM-5) at 5400-6000 tokens vs the rest near 100-300.

**What it shows visually:** clean and striking. Three models are visually separated by an order of magnitude.

**Why I'd skip it:** the appendix already names these three models explicitly in prose ("Three models (DeepSeek R1, GLM-5, Qwen3 Next-80B-A3B-Thinking) emit 5000–6000 tokens per request..."). The same information is in `convergence_matrix.png` (the leftmost column with 5814/5443/6018 cell values). Adding a third figure dedicated to verbosity alone would be redundant — the heatmap already shows the verbosity cluster in context with the other signals.

If the appendix were standalone (not the heatmap-included version), this would be a solid recommendation. With the heatmap included, it's redundant.

### 4. `verbosity_vs_mc_scatter.png` — DO NOT RECOMMEND

A 3-panel small-multiples figure: verbosity vs MC, verbosity vs Task 1 delegation slope, verbosity vs Task 2 coherence.

**What it shows visually:** the verbose-CoT cluster (right side of each panel at ~6000 tokens) sits at zero on all three engagement signals.

**Why I'd skip it:** three panels is a lot of paper real estate for three nearly-identical statistical claims (none of the Pearson/Spearman p-values reach 0.05). The sample is n=23 per panel which means individual scatterplot points are unstable. The story is already conveyed by the heatmap.

### 5. `task2_coherence_scatter.png` — DO NOT RECOMMEND

A 21-panel small-multiples grid showing per-model Task 2 declared-routing coherence scatterplots.

**What it shows visually:** model-by-model scatter of declared MSV activation vs chosen routing action.

**Why I'd skip it:** 21 small panels is a layout disaster for a paper figure (each panel would be ~1cm wide at paper width). The top-level signal (per-model coherence ρ) is already in the convergence matrix. This figure is excellent diagnostic detail for the audit pipeline (and is perfect for `task11_analysis/outputs/`) but doesn't translate to print.

### 6. `delegation_by_difficulty.png` — DO NOT RECOMMEND

A line plot showing Task 1 delegation rate as a function of question difficulty, one line per model.

**What it shows visually:** noisy. Many overlapping lines. A few outliers (gpt-5.4-2026-03-05 at 0.5 delegation rate on easier questions, and the verbose-CoT cluster on the bottom at ~0).

**Why I'd skip it:** the visual story is unclear. There's no obvious cluster pattern. The data underlying it is more cleanly summarized in the t01_delegate_slope column of the convergence matrix.



## Why not the main text

All four current paper figures are in Section 5 (results) or Appendix D. The Task 11 audit story is inherently a "we discovered a preprocessing issue and corrected the cohort-level finding" narrative — it's epistemological detail about how to evaluate metacognition correctly, not a primary result. That argues for appendix placement, not main text.

If main-text figures were under consideration, the story could be: replace the current Figure 4 (completion heatmap) with the convergence matrix, since the convergence matrix is more central to the paper's thesis (cross-task heterogeneity) and the completion heatmap is more of a supplementary finding. But the current Figure 4 is a deliberate choice for Section 5.5, so I wouldn't recommend that swap unless the authors specifically want to elevate the audit story.

