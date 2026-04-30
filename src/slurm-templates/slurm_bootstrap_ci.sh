#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --job-name=msv_boot
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=0-04:00:00
#SBATCH --output=/home/%u/msv_benchmark/kaggle_neurips/logs/boot_%j_%x.out
#SBATCH --error=/home/%u/msv_benchmark/kaggle_neurips/logs/boot_%j_%x.err

set -euo pipefail

PROJECT_DIR="${HOME}/msv_benchmark/kaggle_neurips"

# venv lives one level up at ~/msv_benchmark/.venv/
source "${HOME}/msv_benchmark/.venv/bin/activate"

cd "${PROJECT_DIR}"
mkdir -p results/reproduced/bootstrap logs

# Override at submit time with --export=ALL,N_BOOT=...,INPUT_DIR=...,OUTPUT_DIR=...
N_BOOT="${N_BOOT:-10000}"
INPUT_DIR="${INPUT_DIR:-results/reproduced/analysis_input/delegate_game/}"
OUTPUT_DIR="${OUTPUT_DIR:-results/reproduced/bootstrap/}"

echo "==> Bootstrap CI job"
echo "    n_boot     = ${N_BOOT}"
echo "    input_dir  = ${INPUT_DIR}"
echo "    output_dir = ${OUTPUT_DIR}"
echo "    node       = $(hostname)"
echo "    started    = $(date)"
echo ""

FORCED_ANSWER_DIR="${FORCED_ANSWER_DIR:-}"

if [[ -n "${FORCED_ANSWER_DIR}" ]]; then
    echo "    forced_answer_dir = ${FORCED_ANSWER_DIR}"
    python scripts/compute_bootstrap_ci.py \
        --input_dir         "${INPUT_DIR}" \
        --forced_answer_dir "${FORCED_ANSWER_DIR}" \
        --output_dir        "${OUTPUT_DIR}" \
        --n_boot            "${N_BOOT}"
else
    python scripts/compute_bootstrap_ci.py \
        --input_dir  "${INPUT_DIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --n_boot     "${N_BOOT}"
fi

echo ""
echo "==> Done at $(date)"