#!/bin/bash
#SBATCH --job-name=submit_bulk_seeds.sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

NUM_RUNS=${1:-10}
START_SEED=${2:-1}
BATCH_SIZE=${3:-2}
POLL_SECONDS=${4:-30}
RUN_TAG=${5:-$(date +%F)}
MAX_RETRY_ROUNDS=${6:-4}
RUN_DIR="runs_${RUN_TAG}"
BATCH_ID=${SLURM_JOB_ID:-manual-$(date +%s)}

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch not found. Run this on the cluster login node."
  exit 1
fi

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue not found. Run this on the cluster login node."
  exit 1
fi

mkdir -p "${RUN_DIR}"

echo "Submitting ${NUM_RUNS} runs from seed ${START_SEED} in batches of ${BATCH_SIZE}."
echo "Output directory: ${RUN_DIR}/"
echo "Batch ID tag: ${BATCH_ID}"
echo "Auto-retry rounds: ${MAX_RETRY_ROUNDS}"

submitted=0
current_seed=${START_SEED}
END_SEED=$((START_SEED + NUM_RUNS - 1))

seed_has_summary() {
  local seed="$1"
  local f
  for f in "${RUN_DIR}"/slurm-batch*-seed"${seed}"-*.out; do
    [ -f "${f}" ] || continue
    if grep -q "=== EVALUATION SUMMARY" "${f}"; then
      return 0
    fi
  done
  return 1
}

collect_missing_seeds() {
  local s
  for s in $(seq "${START_SEED}" "${END_SEED}"); do
    if ! seed_has_summary "${s}"; then
      echo "${s}"
    fi
  done
}

wait_for_jobs() {
  local ids=("$@")
  while true; do
    active=0
    for jid in "${ids[@]}"; do
      if squeue -h -j "$jid" | grep -q .; then
        active=1
        break
      fi
    done

    if [ "$active" -eq 0 ]; then
      break
    fi

    sleep "$POLL_SECONDS"
  done
}

while [ "$submitted" -lt "$NUM_RUNS" ]; do
  batch_job_ids=()

  for ((i=0; i<BATCH_SIZE && submitted<NUM_RUNS; i++)); do
    seed=${current_seed}
    job_id=$(sbatch --parsable \
      --export=ALL,EXPERIMENT_SEED=${seed},BATCH_ID=${BATCH_ID} \
      --job-name="llm_seed_${seed}" \
      --output="${RUN_DIR}/slurm-batch${BATCH_ID}-seed${seed}-%j.out" \
      run_job.sh)

    echo "Submitted seed=${seed} job_id=${job_id}"
    batch_job_ids+=("${job_id}")
    submitted=$((submitted + 1))
    current_seed=$((current_seed + 1))
  done

  echo "Waiting for batch to finish: ${batch_job_ids[*]}"
  wait_for_jobs "${batch_job_ids[@]}"
  echo "Batch complete."
done

for ((round=1; round<=MAX_RETRY_ROUNDS; round++)); do
  mapfile -t missing_seeds < <(collect_missing_seeds)

  if [ "${#missing_seeds[@]}" -eq 0 ]; then
    echo "All ${NUM_RUNS} seeds have evaluation summaries."
    break
  fi

  echo "Retry round ${round}/${MAX_RETRY_ROUNDS} for missing seeds: ${missing_seeds[*]}"

  idx=0
  while [ "$idx" -lt "${#missing_seeds[@]}" ]; do
    batch_job_ids=()
    for ((i=0; i<BATCH_SIZE && idx<${#missing_seeds[@]}; i++)); do
      seed=${missing_seeds[$idx]}
      retry_tag="${BATCH_ID}-r${round}"
      job_id=$(sbatch --parsable \
        --export=ALL,EXPERIMENT_SEED=${seed},BATCH_ID=${retry_tag} \
        --job-name="llm_seed_${seed}" \
        --output="${RUN_DIR}/slurm-batch${retry_tag}-seed${seed}-%j.out" \
        run_job.sh)
      echo "Resubmitted seed=${seed} job_id=${job_id}"
      batch_job_ids+=("${job_id}")
      idx=$((idx + 1))
    done

    echo "Waiting for retry batch to finish: ${batch_job_ids[*]}"
    wait_for_jobs "${batch_job_ids[@]}"
    echo "Retry batch complete."
  done
done

mapfile -t final_missing < <(collect_missing_seeds)
if [ "${#final_missing[@]}" -eq 0 ]; then
  echo "Completed: all ${NUM_RUNS} seeds have final summaries."
else
  echo "WARNING: missing summaries after retries for seeds: ${final_missing[*]}"
fi

echo "Collect results with: python3 gather_results.py \"${RUN_DIR}/slurm-batch*-seed*.out\""
