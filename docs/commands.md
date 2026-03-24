# Cluster copy (optional)
scp *filename.py* hendrix:/home/*ku-id*/Dir/
scp -r BA-LLM hendrix:/home/wkg105/Dir/

# Single seeded run (from repo root on cluster login node)
mkdir -p runs_smoke
sbatch --export=ALL,EXPERIMENT_SEED=1 \
  --job-name="llm_seed_1_smoke" \
  --output="runs_smoke/slurm-seed1-%j.out" \
  run_job.sh

# Bulk seeded runs
bash submit_bulk_seeds.sh 10 1 2 30 tool_web5_$(date +%F)

# Parse all run outputs into results CSV + aggregate summary
python3 gather_results.py
