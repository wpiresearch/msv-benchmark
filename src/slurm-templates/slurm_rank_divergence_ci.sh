#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --job-name=msv_rdci
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=0-02:00:00
#SBATCH --output=/home/%u/msv_benchmark/kaggle_neurips/logs/rdci_%j_%x.out
#SBATCH --error=/home/%u/msv_benchmark/kaggle_neurips/logs/rdci_%j_%x.err

set -euo pipefail

PROJECT_DIR="${HOME}/msv_benchmark/kaggle_neurips"
source "${HOME}/msv_benchmark/.venv/bin/activate"
cd "${PROJECT_DIR}"
mkdir -p results/reproduced

N_BOOT="${N_BOOT:-10000}"
INPUT_DIR="${INPUT_DIR:-results/reproduced/analysis_input/delegate_game/}"

echo "==> Rank-divergence bootstrap"
echo "    n_boot     = ${N_BOOT}"
echo "    started    = $(date)"
echo ""

FORCED_ANSWER_DIR="${FORCED_ANSWER_DIR:-}"

echo "==> Primary (own-error AUC, all-computable, n=11)"
if [[ -n "${FORCED_ANSWER_DIR}" ]]; then
    echo "    forced_answer_dir = ${FORCED_ANSWER_DIR}"
    python scripts/compute_rank_divergence_ci.py \
        --input_dir         "${INPUT_DIR}" \
        --forced_answer_dir "${FORCED_ANSWER_DIR}" \
        --n_boot            "${N_BOOT}" \
        --min_answered      5 \
        --output_csv        results/reproduced/rank_divergence_bootstrap.csv
else
    python scripts/compute_rank_divergence_ci.py \
        --input_dir     "${INPUT_DIR}" \
        --n_boot        "${N_BOOT}" \
        --min_answered  5 \
        --output_csv    results/reproduced/rank_divergence_bootstrap.csv
fi

echo ""
echo "==> Sensitivity (n=10, min_answered>=20)"
if [[ -n "${FORCED_ANSWER_DIR}" ]]; then
    python scripts/compute_rank_divergence_ci.py \
        --input_dir         "${INPUT_DIR}" \
        --forced_answer_dir "${FORCED_ANSWER_DIR}" \
        --n_boot            "${N_BOOT}" \
        --min_answered      20 \
        --output_csv        results/reproduced/rank_divergence_bootstrap_n20.csv
else
    python scripts/compute_rank_divergence_ci.py \
        --input_dir     "${INPUT_DIR}" \
        --n_boot        "${N_BOOT}" \
        --min_answered  20 \
        --output_csv    results/reproduced/rank_divergence_bootstrap_n20.csv
fi

echo ""
echo "==> Done at $(date)"
