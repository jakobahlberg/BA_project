#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu            # GPU partition
#SBATCH --gres=gpu:1               # request 1 GPU
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                  # request more memory
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# Load Python and CUDA modules
module load python/3.12.8
module load cuda/11.8              # adjust if needed

# Optional: create a virtual environment
# python3 -m venv venv
# source venv/bin/activate

# Install required Python packages locally (inside user home)
python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker fpdf2 pymupdf sentence-transformers

mkdir -p ~/Dir/carbon_logs


# Run your script
python3 ~/Dir/20_questions_hint.py