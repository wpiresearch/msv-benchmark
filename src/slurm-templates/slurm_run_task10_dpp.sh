#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --job-name=msv_dpp
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=0-12:00:00
#SBATCH --output=/home/%u/msv_benchmark/kaggle_neurips/logs/dpp_%j_%x.out
#SBATCH --error=/home/%u/msv_benchmark/kaggle_neurips/logs/dpp_%j_%x.err

set -euo pipefail

MODEL="${MODEL:?ERROR: MODEL must be set via --export}"
NUM_CTX="${NUM_CTX:-32768}"
PROJECT_DIR="${HOME}/msv_benchmark/kaggle_neurips"
SCRATCH_DIR="/scratch/${USER}"
GPQA_JSONL="${PROJECT_DIR}/data/gpqa_diamond_80.jsonl"

# Pre-flight checks
echo "==> Pre-flight checks"
if [[ ! -f "${GPQA_JSONL}" ]]; then
    echo "ERROR: GPQA JSONL not found at ${GPQA_JSONL}"
    exit 2
fi
if [[ ! -f "${SCRATCH_DIR}/ollama/ollama.sif" ]]; then
    echo "ERROR: Ollama SIF not found at ${SCRATCH_DIR}/ollama/ollama.sif"
    exit 2
fi
echo "    GPQA JSONL: $(wc -l < "${GPQA_JSONL}") lines"
echo "    Ollama SIF: present"
echo "    NUM_CTX: ${NUM_CTX}"

module load apptainer
source "${HOME}/msv_benchmark/.venv/bin/activate"

# Start ollama
mkdir -p "${SCRATCH_DIR}/ollama/models"
OLLAMA_LOG="/tmp/ollama_${SLURM_JOB_ID}.log"
apptainer exec --userns --nv \
    --bind "${SCRATCH_DIR}/ollama/models:${HOME}/.ollama/models" \
    "${SCRATCH_DIR}/ollama/ollama.sif" \
    ollama serve > "${OLLAMA_LOG}" 2>&1 &
OLLAMA_PID=$!

# Wait for ollama
echo "==> Waiting for ollama (PID ${OLLAMA_PID})"
OLLAMA_READY=0
for i in $(seq 1 60); do
    if curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "    ollama ready after ${i}s"
        OLLAMA_READY=1
        break
    fi
    if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
        echo "ERROR: ollama process died before becoming ready"
        tail -50 "${OLLAMA_LOG}"
        exit 3
    fi
    sleep 1
done
[[ "${OLLAMA_READY}" -ne 1 ]] && { echo "ERROR: ollama timeout"; tail -50 "${OLLAMA_LOG}"; kill "${OLLAMA_PID}" 2>/dev/null || true; exit 4; }

# Verify model
if ! curl -s http://localhost:11434/api/tags | grep -q "\"name\":\"${MODEL}\""; then
    echo "ERROR: model ${MODEL} not in ollama. Available models:"
    curl -s http://localhost:11434/api/tags | python -m json.tool
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 5
fi
echo "==> Model ${MODEL} confirmed available"

# Run DPP
cd "${PROJECT_DIR}"
mkdir -p results/results-gpqa-2026-03-25/task10_dpp/

echo "==> Running Task 10 DPP for ${MODEL}"
echo "    started = $(date)"

PYTHON_EXIT=0
python scripts/run_task10_dpp_turing.py \
    --gpqa-jsonl "${GPQA_JSONL}" \
    --models     "${MODEL}" \
    --output-dir results/results-gpqa-2026-03-25/task10_dpp/ \
    --num-ctx    "${NUM_CTX}" \
    || PYTHON_EXIT=$?

# Cleanup
if kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    kill "${OLLAMA_PID}" 2>/dev/null
fi

if [[ "${PYTHON_EXIT}" -ne 0 ]]; then
    echo "ERROR: python script failed with exit code ${PYTHON_EXIT}"
    exit "${PYTHON_EXIT}"
fi

echo "==> Done at $(date)"
