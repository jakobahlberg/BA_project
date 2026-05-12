#!/bin/bash
#SBATCH --job-name=quantize
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=hendrixgpu26fl,hendrixgpu17fl,hendrixgpu18fl
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-quantize-%j.out

module load python/3.12.8
module load cuda/11.8

export CUDA_VISIBLE_DEVICES=0,1
    
nvidia-smi

python3 -m pip install --user transformers accelerate torch --quiet
python3 -m pip install --user git+https://github.com/saintslab/osc_reg.git --quiet

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

# ── Configure here ────────────────────────────────────────────────────────────
MODEL="Qwen/Qwen3-8B"
BITS=8
OUTPUT="$SLURM_SUBMIT_DIR/quantized_models/$(basename $MODEL)-${BITS}bit"
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "$SLURM_SUBMIT_DIR/quantized_models"

python3 "$SLURM_SUBMIT_DIR/quantize_model_exclusions.py" \
    --model  "$MODEL" \
    --bits   "$BITS" \
    --output "$OUTPUT"

echo "Quantized model saved to: $OUTPUT"
echo "Set GUESSER_MODEL = \"$OUTPUT\" in config.py to use it."

#Ratio: 128 quantized / (128 + 250) ≈ 34% of weight-bearing modules quantized, 66% kept in fp16.