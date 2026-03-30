#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3
#SBATCH --exclude=hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu26fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out


module load python/3.12.8
module load cuda/11.8

unset CUDA_VISIBLE_DEVICES
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



# Verify GPU is visible
nvidia-smi
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

# Install dependencies once on login node instead of every job to avoid CUDA re-init issues
# python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker sentence-transformers fpdf2 pymupdf

# HuggingFace cache (override via environment if needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

mkdir -p "$SLURM_SUBMIT_DIR/carbon_logs"

# Run from the directory where sbatch was called — works for any home dir
python3 "$SLURM_SUBMIT_DIR/run.py"
