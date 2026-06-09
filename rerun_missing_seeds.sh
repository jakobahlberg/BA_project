#!/bin/bash
set -euo pipefail

# Auto-resubmit only missing/failed seeds for run folders.
#
# Usage:
#   bash rerun_missing_seeds.sh "<run_dir_glob>" [expected_runs] [start_seed] [batch_size] [poll_seconds] [max_rounds]
#
# Example:
#   bash rerun_missing_seeds.sh "runs_grid_*_2026-04-09_073049" 10 1 2 30 4

RUN_GLOB=${1:-}
EXPECTED_RUNS=${2:-10}
START_SEED=${3:-1}
BATCH_SIZE=${4:-2}
POLL_SECONDS=${5:-30}
MAX_ROUNDS=${6:-4}

if [ -z "${RUN_GLOB}" ]; then
  echo "Usage: bash rerun_missing_seeds.sh \"<run_dir_glob>\" [expected_runs] [start_seed] [batch_size] [poll_seconds] [max_rounds]"
  exit 1
fi

if ! command -v sbatch >/dev/null 2>&1 || ! command -v squeue >/dev/null 2>&1; then
  echo "This script must run on a SLURM login node (sbatch/squeue required)."
  exit 1
fi

expand_dirs() {
  shopt -s nullglob
  local dirs=( $RUN_GLOB )
  shopt -u nullglob
  printf '%s\n' "${dirs[@]}"
}

detect_mode() {
  local dir="$1"
  local first
  first=$(ls -1 "$dir"/slurm-batch*-seed*.out 2>/dev/null | head -n 1 || true)
  if [ -n "$first" ] && grep -q "TOOL MODE" "$first"; then
    echo "tool"
  else
    echo "standard"
  fi
}

detect_models() {
  local dir="$1"
  local first
  first=$(ls -1 "$dir"/slurm-batch*-seed*.out 2>/dev/null | head -n 1 || true)

  local guesser secret
  guesser=$(grep -m1 "\\[Models\\] Loading:" "$first" 2>/dev/null | sed 's/.*Loading: //')
  secret=$(grep "\\[Models\\] Loading:" "$first" 2>/dev/null | sed 's/.*Loading: //' | sed -n '2p')

  if [ -z "${guesser}" ] || [ -z "${secret}" ]; then
    return 1
  fi

  echo "${guesser}|${secret}"
}

successful_seeds() {
  local dir="$1"
  local f seed

  for f in "$dir"/slurm-batch*-seed*.out; do
    [ -f "$f" ] || continue
    grep -q "=== EVALUATION SUMMARY" "$f" || continue
    seed=$(grep -m1 "experiment_seed:" "$f" | sed -E 's/.*experiment_seed:[[:space:]]*([0-9]+).*/\1/' || true)
    if [ -z "$seed" ]; then
      seed=$(basename "$f" | sed -E 's/.*seed([0-9]+)-.*/\1/')
    fi
    [ -n "$seed" ] && echo "$seed"
  done | sort -n | uniq
}

missing_seeds() {
  local dir="$1"
  local end_seed=$((START_SEED + EXPECTED_RUNS - 1))
  local have
  have=$(successful_seeds "$dir" | tr '\n' ' ')

  local s
  for s in $(seq "$START_SEED" "$end_seed"); do
    if ! grep -qw "$s" <<< "$have"; then
      echo "$s"
    fi
  done
}

wait_for_jobs() {
  local ids=("$@")
  while true; do
    local active=0
    local jid
    for jid in "${ids[@]}"; do
      if squeue -h -j "$jid" | grep -q .; then
        active=1
        break
      fi
    done
    [ "$active" -eq 0 ] && break
    sleep "$POLL_SECONDS"
  done
}

submit_missing_for_dir() {
  local dir="$1"

  local mode
  mode=$(detect_mode "$dir")

  local models_line
  if ! models_line=$(detect_models "$dir"); then
    echo "  ! Could not detect models from ${dir}; skipping."
    return
  fi

  local guesser secret
  guesser=$(echo "$models_line" | cut -d'|' -f1)
  secret=$(echo "$models_line" | cut -d'|' -f2)

  local round
  for round in $(seq 1 "$MAX_ROUNDS"); do
    mapfile -t miss < <(missing_seeds "$dir")
    local n_miss=${#miss[@]}
    if [ "$n_miss" -eq 0 ]; then
      echo "  ✓ Complete (${EXPECTED_RUNS}/${EXPECTED_RUNS})"
      return
    fi

    echo "  Round ${round}/${MAX_ROUNDS}: missing seeds -> ${miss[*]}"
    local i=0
    while [ "$i" -lt "$n_miss" ]; do
      local batch_ids=()
      local j
      for j in $(seq 1 "$BATCH_SIZE"); do
        [ "$i" -ge "$n_miss" ] && break
        local seed="${miss[$i]}"
        local rerun_tag="rerun-$(basename "$dir")-r${round}-$(date +%Y%m%d%H%M%S)"
        local jid
        jid=$(sbatch --parsable \
          --export=ALL,MODE="${mode}",GUESSER_MODEL="${guesser}",SECRET_MODEL="${secret}",EXPERIMENT_SEED="${seed}",BATCH_ID="${rerun_tag}" \
          --job-name="llm_seed_${seed}" \
          --output="${dir}/slurm-batch${rerun_tag}-seed${seed}-%j.out" \
          run_job.sh)
        echo "    submitted seed=${seed} job_id=${jid}"
        batch_ids+=("$jid")
        i=$((i + 1))
      done

      echo "    waiting for batch: ${batch_ids[*]}"
      wait_for_jobs "${batch_ids[@]}"
    done
  done

  mapfile -t miss_final < <(missing_seeds "$dir")
  if [ "${#miss_final[@]}" -eq 0 ]; then
    echo "  ✓ Complete (${EXPECTED_RUNS}/${EXPECTED_RUNS})"
  else
    echo "  ! Still missing after ${MAX_ROUNDS} rounds: ${miss_final[*]}"
  fi
}

main() {
  mapfile -t dirs < <(expand_dirs)
  if [ "${#dirs[@]}" -eq 0 ]; then
    echo "No run directories matched: ${RUN_GLOB}"
    exit 1
  fi

  echo "Matched ${#dirs[@]} run directory(ies)."
  echo "expected_runs=${EXPECTED_RUNS}, start_seed=${START_SEED}, batch_size=${BATCH_SIZE}, poll_seconds=${POLL_SECONDS}, max_rounds=${MAX_ROUNDS}"

  local d
  for d in "${dirs[@]}"; do
    echo ""
    echo "== ${d} =="
    submit_missing_for_dir "$d"
  done
}

main
