#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=hendrixgpu26fl,hendrixgpu17fl,hendrixgpu18fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --include=hendrixgpu13fl

module load python/3.12.8
module load cuda/11.8

export CUDA_VISIBLE_DEVICES=0,1

# Verify GPU is visible
nvidia-smi
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

# Install dependencies (safe to re-run; skips if already installed)
python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker sentence-transformers fpdf2 pymupdf

# HuggingFace cache (override via environment if needed)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

mkdir -p "$SLURM_SUBMIT_DIR/carbon_logs"

# Run from the directory where sbatch was called — works for any home dir
python3 "$SLURM_SUBMIT_DIR/run.py"
