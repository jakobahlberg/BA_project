#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=hendrixgpu01fl,hendrixfut01fl,hendrixgpu03fl,hendrixgpu04fl,hendrixgpu06fl,hendrixgpu07fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu12fl,hendrixgpu13fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu20fl,hendrixgpu22fl,hendrixgpu23fl,hendrixgpu26fl

set -euo pipefail

module load python/3.12.8
module load cuda/11.8

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Verify GPU is visible
nvidia-smi
python3 - <<'PY'
import os
import sys
import torch

print("CUDA_VISIBLE_DEVICES:", os.getenv("CUDA_VISIBLE_DEVICES"))
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    print("FATAL: CUDA not usable in this allocation", file=sys.stderr)
    sys.exit(42)

# Some nodes pass torch.cuda.is_available() but still fail on first real kernels
# with "no kernel image is available for execution on the device".
for idx in range(torch.cuda.device_count()):
    try:
        with torch.cuda.device(idx):
            x = torch.randn((256, 256), device=f"cuda:{idx}", dtype=torch.float16)
            y = x @ x
            torch.cuda.synchronize(idx)
            _ = float(y[0, 0].item())
        print(f"CUDA kernel smoke test ok on cuda:{idx}")
    except Exception as e:
        print(f"FATAL: CUDA kernel smoke test failed on cuda:{idx}: {e}", file=sys.stderr)
        sys.exit(43)
PY

# Install dependencies once on login node instead of every job to avoid CUDA re-init issues
# python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker sentence-transformers fpdf2 pymupdf ddgs

# HuggingFace cache (override via environment if needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# HuggingFace token — required for gated models (e.g. Gemma).
# Reads the token saved by `hf auth login` on the login node.
if [ -z "${HF_TOKEN:-}" ]; then
    _TOKEN_FILE="${HF_HOME}/token"
    if [ -f "${_TOKEN_FILE}" ]; then
        export HF_TOKEN=$(cat "${_TOKEN_FILE}")
        echo "HF_TOKEN loaded from ${_TOKEN_FILE}"
    else
        export HF_TOKEN="hf_REPLACE_WITH_YOUR_TOKEN"
    fi
fi

export CARBON_LOG_DIR="${SLURM_TMPDIR:-/tmp}/carbon_logs_${SLURM_JOB_ID:-$$}"
mkdir -p "$CARBON_LOG_DIR"

# Run from the directory where sbatch was called — works for any home dir
python3 "$SLURM_SUBMIT_DIR/run.py"
