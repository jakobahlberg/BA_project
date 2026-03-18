#!/bin/bash
set -euo pipefail

NUM_RUNS=${1:-10}
START_SEED=${2:-1}
BATCH_SIZE=${3:-2}
POLL_SECONDS=${4:-30}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on the cluster login node."
  exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found. Run this on the cluster login node."
  exit 1
fi

echo "Submitting ${NUM_RUNS} runs from seed ${START_SEED} in batches of ${BATCH_SIZE}."

submitted=0
current_seed=${START_SEED}

while [ "$submitted" -lt "$NUM_RUNS" ]; do
  batch_job_ids=()

  for ((i=0; i<BATCH_SIZE && submitted<NUM_RUNS; i++)); do
    seed=${current_seed}
    job_id=$(sbatch --parsable \
      --export=ALL,EXPERIMENT_SEED=${seed} \
      --job-name="llm_seed_${seed}" \
      --output="slurm-seed${seed}-%j.out" \
      run.sh)

    echo "Submitted seed=${seed} job_id=${job_id}"
    batch_job_ids+=("${job_id}")
    submitted=$((submitted + 1))
    current_seed=$((current_seed + 1))
  done

  echo "Waiting for batch to finish: ${batch_job_ids[*]}"

  while true; do
    active=0
    for jid in "${batch_job_ids[@]}"; do
      if squeue -h -j "$jid" | grep -q .; then
        active=1
        break
      fi
    done

    if [ "$active" -eq 0 ]; then
      echo "Batch complete."
      break
    fi

    sleep "$POLL_SECONDS"
  done
done

echo "All ${NUM_RUNS} seeded runs submitted and completed (or left queue)."
