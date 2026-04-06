#!/bin/bash
set -euo pipefail

# Usage:
#   bash submit_model_grid.sh [config_csv] [num_runs] [start_seed] [batch_size] [poll_seconds] [tag_prefix]
#
# Example:
#   bash submit_model_grid.sh model_configs.csv 10 1 2 30 grid
#
# The script supports two CSV row formats:
#   1) Explicit configuration row:
#      label,mode,guesser_model,secret_model,judge_model
#   2) Model-pool row (auto cartesian expansion):
#      label,mode,models,judge_model
#      where models is semicolon-separated, e.g.
#      Qwen/Qwen3.5-4B-Base;Qwen/Qwen3-8B
#      This expands to all guesser x secret combinations.

CONFIG_CSV=${1:-model_configs.csv}
NUM_RUNS=${2:-10}
START_SEED=${3:-1}
BATCH_SIZE=${4:-2}
POLL_SECONDS=${5:-30}
TAG_PREFIX=${6:-grid}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on the cluster login node."
  exit 1
fi

if [ ! -f "${CONFIG_CSV}" ]; then
  echo "Config CSV not found: ${CONFIG_CSV}"
  exit 1
fi

echo "Submitting model grid from ${CONFIG_CSV}"
echo "Per config: runs=${NUM_RUNS}, start_seed=${START_SEED}, batch_size=${BATCH_SIZE}, poll_seconds=${POLL_SECONDS}"

while IFS=',' read -r label mode guesser secret judge; do
  # Skip header / empty / comments
  if [ -z "${label}" ] || [[ "${label}" = \#* ]]; then
    continue
  fi

  # Header handling:
  # - explicit format header: label,mode,guesser_model,secret_model,judge_model
  # - pool format header:     label,mode,models,judge_model
  if [ "${label}" = "label" ]; then
    continue
  fi

  # Pool row if 4th field is empty: label,mode,models,judge_model
  if [ -n "${label}" ] && [ -n "${mode}" ] && [ -n "${guesser}" ] && [ -z "${judge}" ]; then
    models_csv="${guesser}"
    judge_model="${secret}"

    IFS=';' read -r -a model_list <<< "${models_csv}"
    for gm in "${model_list[@]}"; do
      gm="$(echo "${gm}" | xargs)"
      [ -z "${gm}" ] && continue
      for sm in "${model_list[@]}"; do
        sm="$(echo "${sm}" | xargs)"
        [ -z "${sm}" ] && continue

        run_label="${label}_g$(basename "${gm}")_s$(basename "${sm}")"
        run_tag="${TAG_PREFIX}_${run_label}_$(date +%F_%H%M%S)"
        wrapper_job_id=$(sbatch --parsable \
          --export=ALL,MODE="${mode}",GUESSER_MODEL="${gm}",SECRET_MODEL="${sm}",JUDGE_MODEL="${judge_model}" \
          submit_bulk_seeds.sh "${NUM_RUNS}" "${START_SEED}" "${BATCH_SIZE}" "${POLL_SECONDS}" "${run_tag}")

        echo "Submitted config=${run_label} wrapper_job_id=${wrapper_job_id} run_tag=${run_tag}"
      done
    done
    continue
  fi

  # Explicit row: label,mode,guesser_model,secret_model,judge_model
  run_tag="${TAG_PREFIX}_${label}_$(date +%F_%H%M%S)"
  wrapper_job_id=$(sbatch --parsable \
    --export=ALL,MODE="${mode}",GUESSER_MODEL="${guesser}",SECRET_MODEL="${secret}",JUDGE_MODEL="${judge}" \
    submit_bulk_seeds.sh "${NUM_RUNS}" "${START_SEED}" "${BATCH_SIZE}" "${POLL_SECONDS}" "${run_tag}")

  echo "Submitted config=${label} wrapper_job_id=${wrapper_job_id} run_tag=${run_tag}"
done < "${CONFIG_CSV}"
