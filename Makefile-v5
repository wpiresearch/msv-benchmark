# MSV Benchmark -- Reproducibility Workflow
#
# Usage:
#   make setup              -- install Python dependencies
#   make tests              -- run consistency tests (must pass before trusting numbers)
#   make reproduce-all      -- run everything: stats + figures + Task 10 DPP + Task 11 audit
#   make reproduce-kaggle   -- end-to-end Kaggle cohort pipeline (raw -> extracted -> analysis)
#   make reproduce-figures  -- regenerate all 6 paper figures
#   make reproduce-stats    -- bootstrap CIs + Cronbach alpha + rank divergence
#   make reproduce-task10-dpp -- Task 10 DPP analysis (requires compute_task10_dpp_analysis.py)
#   make reproduce-task11   -- Task 11 audit pipeline (cross-task convergence + Haiku correction)
#   make reproduce-turing   -- driver commands for Turing HPC re-runs (requires ollama)
#   make validate-croissant -- validate both Kaggle and Turing Croissant metadata
#   make anonymize-check    -- scan package for identity / provenance leaks
#   make clean              -- remove generated outputs (leaves raw data intact)
#
# Outputs are written under ./results/reproduced/ to keep the bundled
# ./results/ authoritative and intact.
#
# To reduce bootstrap iteration count for development (default 10000):
#   make reproduce-stats BOOT=500

PY               := python3
REPRO_DIR        := results/reproduced
KAGGLE_EXTRACTED := data/kaggle-data/kaggle_extracted
KAGGLE_RAW       := data/kaggle-data/kaggle_raw
SRC_DIST         := src/distribution
SRC_TESTS        := src/tests
TASK11_ROOT      := src/task11-notes-writeup-experiments

.PHONY: setup tests reproduce-all reproduce-kaggle reproduce-figures reproduce-stats \
        reproduce-task10-dpp reproduce-task11 reproduce-turing \
        validate-croissant anonymize-check clean help

help:
	@grep '^# ' Makefile | sed 's/^# //'

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
setup:
	$(PY) -m pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Consistency tests (run first; must pass before trusting downstream numbers)
# ---------------------------------------------------------------------------
tests:
	@echo ">>> Running consistency tests..."
	$(PY) $(SRC_TESTS)/test_ece_consistency.py
	$(PY) $(SRC_TESTS)/test_hardness_auc_consistency.py
	@echo ">>> All tests passed."

# ---------------------------------------------------------------------------
# Aggregate target: reproduce everything
# ---------------------------------------------------------------------------
reproduce-all: tests reproduce-stats reproduce-figures reproduce-task10-dpp reproduce-task11
	@echo ""
	@echo ">>> All paper-bound numbers reproduced under $(REPRO_DIR)/"

# ---------------------------------------------------------------------------
# End-to-end Kaggle cohort pipeline
#
# Runs in about 2 minutes on a single core. Reproduces every finding
# attributable to the 23-model Kaggle cohort.
# ---------------------------------------------------------------------------
reproduce-kaggle: $(REPRO_DIR)
	@echo ">>> Stage 1: extract raw Kaggle archives"
	unzip -o $(KAGGLE_RAW)/outputs_logs_corrected.zip -d $(REPRO_DIR)/raw/
	$(PY) $(SRC_DIST)/extract_kaggle_outputs.py \
	    --input-dir  $(REPRO_DIR)/raw/outputs_logs/ \
	    --output-dir $(REPRO_DIR)/kaggle_extracted/ \
	    --leaderboard $(KAGGLE_EXTRACTED)/leaderboard_reconciled.csv
	@echo ">>> Stage 2: comparative analysis"
	$(PY) $(SRC_DIST)/analyze_kaggle_cohort.py \
	    --extracted-dir $(REPRO_DIR)/kaggle_extracted/ \
	    --output-dir    $(REPRO_DIR)/kaggle_analysis/ \
	    --n-splits      1000
	@echo ">>> Done. Outputs in $(REPRO_DIR)/kaggle_analysis/"

# ---------------------------------------------------------------------------
# Figures (all 6 paper figures from bundled CSVs)
# ---------------------------------------------------------------------------
reproduce-figures: $(REPRO_DIR)
	@echo ">>> Figure 1, top panels: Rank-reversal scatter (Section 5.2)"
	$(PY) $(SRC_DIST)/generate_rank_reversal_figure.py \
	    --kaggle-csv    results/kaggle_cohort/comparative/delegate_game_metrics.csv \
	    --output-prefix $(REPRO_DIR)/rank_reversal_scatter
	@echo ""
	@echo ">>> Figure 1, bottom panel: ECE vs. Delegation AUC raw-value scatter (Section 5.2, fig:ece_vs_delegauc)"
	$(PY) $(SRC_DIST)/generate_ece_vs_delegauc_scatter.py \
	    --kaggle-csv  results/kaggle_cohort/comparative/delegate_game_metrics.csv \
	    --output-png  $(REPRO_DIR)/ece_vs_delegauc_scatter.png \
	    --output-pdf  $(REPRO_DIR)/ece_vs_delegauc_scatter.pdf \
	    --output-csv  $(REPRO_DIR)/ece_vs_delegauc_scatter.csv
	@echo ""
	@echo ">>> Figure 2: Risk-coverage curves (Section 5.2)"
	$(PY) $(SRC_DIST)/generate_risk_coverage_figure.py \
	    --task1-csv     $(KAGGLE_EXTRACTED)/per_task/t01_delegate_game.csv \
	    --output-prefix $(REPRO_DIR)/risk_coverage
	@echo ""
	@echo ">>> Figure 3: Type-2 AUC vs object-level discrimination (Appendix Task 11 audit)"
	@echo "    (regenerated by 'make reproduce-task11' -- see that target)"
	@echo ""
	@echo ">>> Figure 4: Cross-task convergence heatmap (Appendix Task 11 audit)"
	@echo "    (regenerated by 'make reproduce-task11' -- see that target)"
	@echo ""
	@echo ">>> Figure 5: MSV dimension-subset routing (Appendix D)"
	$(PY) $(SRC_DIST)/generate_msv_routing_subset_analysis.py \
	    --task2-csv    $(KAGGLE_EXTRACTED)/per_task/t02_declared_probe.csv \
	    --metadata-csv $(KAGGLE_EXTRACTED)/run_metadata.csv \
	    --output-prefix $(REPRO_DIR)/msv_routing_subset_analysis
	@echo ""
	@echo ">>> Figure 6: Completion x task heatmap (Appendix D)"
	$(PY) $(SRC_DIST)/generate_completion_heatmap.py \
	    --metadata-csv  $(KAGGLE_EXTRACTED)/run_metadata.csv \
	    --output-prefix $(REPRO_DIR)/completion_heatmap
	@echo ""
	@echo ">>> 5 figures regenerated (Fig 1 top, Fig 1 bottom, Fig 2, Fig 5, Fig 6); 2 Task 11 figures via 'make reproduce-task11'"

# ---------------------------------------------------------------------------
# Statistical analyses
# ---------------------------------------------------------------------------
reproduce-stats: $(REPRO_DIR)
	@echo ">>> Cronbach's alpha on Task 4 (single-task, backward-compatible)"
	$(PY) $(SRC_DIST)/compute_cronbach_alpha_task4.py \
	    --task4-csv $(KAGGLE_EXTRACTED)/per_task/t04_confidence_entropy.csv \
	    --task2-csv $(KAGGLE_EXTRACTED)/per_task/t02_declared_probe.csv \
	    --output-prefix $(REPRO_DIR)/cronbach_alpha_task4
	@echo ">>> Cronbach's alpha across all 11 tasks"
	$(PY) $(SRC_DIST)/compute_cronbach_alpha_all_tasks.py \
	    --per-task-dir $(KAGGLE_EXTRACTED)/per_task/ \
	    --output-prefix $(REPRO_DIR)/cronbach_alpha_all_tasks
	@echo ">>> Adapt Kaggle data for bootstrap pipeline"
	$(PY) $(SRC_DIST)/adapt_kaggle_data.py \
	    --extracted_dir $(KAGGLE_EXTRACTED) \
	    --output_dir    $(REPRO_DIR)/analysis_input/
	@echo ">>> Bootstrap CIs on per-model metrics (n_boot=10000; set BOOT=N for fewer)"
	$(PY) $(SRC_DIST)/compute_bootstrap_ci.py \
	    --input_dir $(REPRO_DIR)/analysis_input/delegate_game/ \
	    --output_dir $(REPRO_DIR)/bootstrap/ \
	    --n_boot $${BOOT:-10000}
	@echo ">>> Bootstrap CIs on cross-model rank divergence (own-error AUC, primary)"
	$(PY) $(SRC_DIST)/compute_rank_divergence_ci.py \
	    --input_dir     $(REPRO_DIR)/analysis_input/delegate_game/ \
	    --auc_target    own_error \
	    --n_boot        $${BOOT:-10000} \
	    --min_answered  5 \
	    --output_csv    $(REPRO_DIR)/rank_divergence_bootstrap.csv
	@echo ">>> Rank-divergence sensitivity (hardness AUC, alternative target)"
	$(PY) $(SRC_DIST)/compute_rank_divergence_ci.py \
	    --input_dir     $(REPRO_DIR)/analysis_input/delegate_game/ \
	    --auc_target    hardness \
	    --n_boot        $${BOOT:-10000} \
	    --min_answered  5 \
	    --output_csv    $(REPRO_DIR)/rank_divergence_hardness.csv
	@echo ">>> Rank-divergence sensitivity (stricter subset, n=10)"
	$(PY) $(SRC_DIST)/compute_rank_divergence_ci.py \
	    --input_dir     $(REPRO_DIR)/analysis_input/delegate_game/ \
	    --n_boot        $${BOOT:-10000} \
	    --min_answered  20 \
	    --output_csv    $(REPRO_DIR)/rank_divergence_bootstrap_n20.csv
	@echo ">>> Stats outputs in $(REPRO_DIR)/"

# ---------------------------------------------------------------------------
# Task 10 DPP analysis (Appendix app:task10_dpp_institutional)
#
# Reads bundled DPP CSVs and FA Phase 1 CSVs; computes per-model matched
# lift, win/loss decomposition, Expert->Generalist trace correction.
# Wall-clock: ~5 minutes on a single CPU.
# ---------------------------------------------------------------------------
reproduce-task10-dpp: $(REPRO_DIR)
	@echo ">>> Task 10 DPP institutional analysis"
	@if [ ! -f $(SRC_DIST)/compute_task10_dpp_analysis.py ]; then \
	    echo "ERROR: compute_task10_dpp_analysis.py missing from $(SRC_DIST)/"; \
	    echo "       The bundled outputs at results/task10_dpp/ remain authoritative."; \
	    exit 1; \
	fi
	$(PY) $(SRC_DIST)/compute_task10_dpp_analysis.py \
	    --dpp-dir         data/task10_dpp/ \
	    --fa-dir          data/forced_answer_phase1/ \
	    --difficulty-csv  data/gpqa_difficulty_scores.csv \
	    --output-dir      $(REPRO_DIR)/task10_dpp/ \
	    --n-boot          $${BOOT:-10000} \
	    --seed            42
	@echo ">>> Task 10 minimum detectable effect (MDE) per model"
	$(PY) $(SRC_DIST)/compute_task10_mde.py \
	    --dpp-dir    data/task10_dpp/ \
	    --fa-dir     data/forced_answer_phase1/ \
	    --output-csv $(REPRO_DIR)/task10_dpp/task10_mde.csv
	@echo ">>> Task 10 DPP outputs in $(REPRO_DIR)/task10_dpp/"

# ---------------------------------------------------------------------------
# Task 11 audit pipeline (Appendix app:task11_audit)
#
# Reads Kaggle data; produces:
#   - Cross-task convergence matrix (figure)
#   - Type-2 AUC vs object-level discrimination scatter (figure)
#   - Per-model Type-2 AUC, MC, d-hat (CSV)
#   - Verbosity stats, delegation curves, Task 2 coherence (CSVs)
# Wall-clock: ~10 minutes on a single CPU.
# ---------------------------------------------------------------------------
reproduce-task11: $(REPRO_DIR)
	@echo ">>> Task 11 audit pipeline (raw-confidence recovery + cross-task convergence)"
	mkdir -p $(REPRO_DIR)/task11_audit/
	cd $(TASK11_ROOT)/task11_analysis && \
	$(PY) ../src-msv-analysis/run_all.py \
	    --data-root   ../../../data/kaggle-data \
	    --output-root ../../../$(REPRO_DIR)/task11_audit/
	@echo ">>> Copying canonical figures to $(REPRO_DIR)/"
	cp $(REPRO_DIR)/task11_audit/convergence_matrix.png  $(REPRO_DIR)/convergence_matrix.png
	cp $(REPRO_DIR)/task11_audit/d_hat_vs_type2auc.png   $(REPRO_DIR)/d_hat_vs_type2auc.png
	@echo ">>> Task 11 audit outputs in $(REPRO_DIR)/task11_audit/"

# ---------------------------------------------------------------------------
# Turing HPC re-runs (requires ollama server + model tags pulled locally)
#
# These commands are documentation-only; they print the actual sbatch
# commands you would run on a Turing-class cluster. The output of these
# runs is already bundled at data/forced_answer_phase1/ and data/task10_dpp/.
# ---------------------------------------------------------------------------
reproduce-turing:
	@echo ">>> Turing re-runs require a local ollama server and the relevant"
	@echo ">>> model tags pulled (llama3.1:8b, llama3.2:3b, llama3.2:1b,"
	@echo ">>> qwen2.5:7b, qwen2.5:3b, phi4-mini:latest, gemma2:9b, gemma2:2b, mistral:7b)."
	@echo ">>> See REPRODUCIBILITY_GUIDE.md Section 6 for full instructions."
	@echo ""
	@echo ">>> The output of both Turing runs is already bundled:"
	@echo "     data/forced_answer_phase1/   (9 FA CSVs)"
	@echo "     data/task10_dpp/             (9 DPP CSVs + 720 transcripts)"
	@echo ""
	@echo ">>> To re-run forced-answer Phase 1 (~2 hours on a single GPU per model):"
	@echo "    sbatch --export=ALL,MODEL=qwen2.5:7b src/slurm-templates/slurm_run_forced_answer.sh"
	@echo ""
	@echo ">>> To re-run Task 10 DPP at extended context (~50 min/model on single A100):"
	@echo "    sbatch --export=ALL,MODEL=qwen2.5:7b,NUM_CTX=32768 src/slurm-templates/slurm_run_task10_dpp.sh"

# ---------------------------------------------------------------------------
# Croissant metadata validation (both Kaggle and Turing cards)
# ---------------------------------------------------------------------------
validate-croissant:
	@echo ">>> Validating data/kaggle-data/croissant_metadata.json (Kaggle cohort)"
	@if $(PY) -c "import mlcroissant" 2>/dev/null; then \
	    $(PY) -m mlcroissant.scripts.validate \
	        --jsonld data/kaggle-data/croissant_metadata.json; \
	else \
	    echo "mlcroissant not installed; install with: pip install mlcroissant"; \
	    exit 1; \
	fi
	@if [ -f data/turing-msv-benchmark-data-metadata.json ]; then \
	    echo ">>> Validating data/turing-msv-benchmark-data-metadata.json (Turing cohort)"; \
	    $(PY) -m mlcroissant.scripts.validate \
	        --jsonld data/turing-msv-benchmark-data-metadata.json; \
	else \
	    echo ">>> data/turing-msv-benchmark-data-metadata.json not yet present (will be added before ship)"; \
	fi

# ---------------------------------------------------------------------------
# Anonymity check: scan the package for identity / provenance leaks
# ---------------------------------------------------------------------------
anonymize-check:
	@echo ">>> Scanning for identity leaks..."
	@! grep -rniE "ricky|sethi|fitchburg|worcester|polytechnic|WPI|hqiu|mfahmy|professorsethi|CharlesPDX|courchaine" . \
	    --include="*.tex" --include="*.md" --include="*.py" --include="*.ipynb" \
	    --include="*.json" --include="*.txt" \
	    2>/dev/null | grep -v 'cite{sethi\|CourchaineSethi\|sethi20\|authors are welcome'
	@echo ">>> No author-identifying strings found."
	@echo ""
	@echo ">>> Scanning for Kaggle paths with usernames..."
	@! grep -rniE "/kaggle/working/|/kaggle/input/[a-z]+sethi|/kaggle/input/professor" . \
	    --include="*.py" --include="*.ipynb" 2>/dev/null
	@echo ">>> No leaky Kaggle paths found."
	@echo ""
	@echo ">>> Scanning for OS metadata / editor temp files..."
	@! find . \( -name ".DS_Store" -o -name "__MACOSX" -o -name ".texpadtmp" \
	          -o -name "._*" -o -name "*.bak" -o -name "*~" \) 2>/dev/null | grep .
	@echo ">>> No OS / editor detritus found."
	@echo ""
	@echo ">>> Scanning for internal provenance files..."
	@! find . \( -iname "*TURN*LOG*" -o -iname "*internal-notes*" -o -iname "claude-turns*" \
	           -o -iname "INTERNAL-*" \) 2>/dev/null | grep .
	@echo ">>> No internal provenance found. Package is anonymity-clean."

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
$(REPRO_DIR):
	mkdir -p $(REPRO_DIR)

clean:
	rm -rf $(REPRO_DIR)
