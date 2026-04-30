#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --job-name=msv_fa
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=0-01:00:00
#SBATCH --output=/home/%u/msv_benchmark/kaggle_neurips/logs/fa_%j_%x.out
#SBATCH --error=/home/%u/msv_benchmark/kaggle_neurips/logs/fa_%j_%x.err

set -euo pipefail

MODEL="${MODEL:?ERROR: MODEL must be set via --export}"
PROJECT_DIR="${HOME}/msv_benchmark/kaggle_neurips"
SCRATCH_DIR="/scratch/${USER}"
GPQA_JSONL="${PROJECT_DIR}/data/gpqa_diamond_80.jsonl"

# Pre-flight checks BEFORE doing any expensive setup
echo "==> Pre-flight checks"
if [[ ! -f "${GPQA_JSONL}" ]]; then
    echo "ERROR: GPQA JSONL not found at ${GPQA_JSONL}"
    echo "Hint: build it with scripts/make_gpqa_jsonl.py"
    exit 2
fi
if [[ ! -f "${SCRATCH_DIR}/ollama/ollama.sif" ]]; then
    echo "ERROR: Ollama SIF not found at ${SCRATCH_DIR}/ollama/ollama.sif"
    echo "Hint: run setup_turing.sh on a login node"
    exit 2
fi
echo "    GPQA JSONL: $(wc -l < "${GPQA_JSONL}") lines"
echo "    Ollama SIF: present"

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

# Wait for ollama to be ready, with a hard timeout
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
        echo "Last 50 lines of ollama log:"
        tail -50 "${OLLAMA_LOG}"
        exit 3
    fi
    sleep 1
done
if [[ "${OLLAMA_READY}" -ne 1 ]]; then
    echo "ERROR: ollama did not become ready within 60s"
    tail -50 "${OLLAMA_LOG}"
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 4
fi

# Verify the model is pulled
if ! curl -s http://localhost:11434/api/tags | grep -q "\"name\":\"${MODEL}\""; then
    echo "ERROR: model ${MODEL} not in ollama. Available models:"
    curl -s http://localhost:11434/api/tags | python -m json.tool
    kill "${OLLAMA_PID}" 2>/dev/null || true
    exit 5
fi
echo "==> Model ${MODEL} confirmed available"

# Run the forced-answer script
cd "${PROJECT_DIR}"
mkdir -p results/results-gpqa-2026-03-25/forced_answer_phase1/

echo "==> Running forced-answer for ${MODEL}"
echo "    started = $(date)"

PYTHON_EXIT=0
python scripts/run_forced_answer_phase1_turing.py \
    --gpqa-jsonl "${GPQA_JSONL}" \
    --models     "${MODEL}" \
    --output-dir results/results-gpqa-2026-03-25/forced_answer_phase1/ \
    || PYTHON_EXIT=$?

# Cleanup ollama (don't mask exit code)
if kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    kill "${OLLAMA_PID}" 2>/dev/null
fi

if [[ "${PYTHON_EXIT}" -ne 0 ]]; then
    echo "ERROR: python script failed with exit code ${PYTHON_EXIT}"
    exit "${PYTHON_EXIT}"
fi

echo "==> Done at $(date)"
