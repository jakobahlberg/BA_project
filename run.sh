#!/bin/bash
#SBATCH --job-name=llm_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --exclude=hendrixgpu26fl,hendrixgpu17fl,hendrixgpu13fl

module load python/3.12.8
module load cuda/11.8

python3 -m pip install --user transformers accelerate torch sentencepiece carbontracker sentence-transformers fpdf2 pymupdf

mkdir -p ~/Dir/carbon_logs

python3 ~/Dir/BA-llm/run.py