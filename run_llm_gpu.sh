#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu            # GPU partition
#SBATCH --gres=gpu:2               # request 1 GPU
#SBATCH --exclude=hendrixgpu26fl,hendrixgpu17fl,hendrixgpu18fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                  # request more memory
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Load Python and CUDA modules
module load python/3.12.8
module load cuda/11.8              # adjust if needed

export CUDA_VISIBLE_DEVICES=0,1
nvidia-smi
python3 - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
PY

# Optional: create a virtual environment
# python3 -m venv venv
# source venv/bin/activate

# Install required Python packages locally (inside user home)

# Hugging Face cache + token (set HF_TOKEN in your environment or here)
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker sentence-transformers

mkdir -p ~/BAdir/carbon_logs


# Run your script
python3 ~/BAdir/llm.py