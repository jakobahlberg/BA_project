# Does Size Predict Behaviour?

Tool Use in Small Language Model Agents

By Lasse Abildhauge Christensen (`blm470`) and Jakob Frederik Ahlberg (`wkg105`).

Supervisor: Raghavendra Selvan

## Overview

This repository contains the code and results for our bachelor project on small language models in agentic settings. We evaluate language models in a controlled `20 Questions`-style game where one model acts as a guesser and another model acts as a secret keeper.

The guesser must ask questions, use tools, and eventually guess the hidden secret. In tool mode, the guesser can use:

- `USE_HINT`
- `WEB_SEARCH`

The project studies whether model size predicts agentic behaviour and task success. We evaluate models not only by win rate, but also by tool use, guesses, questions, keeper reliability, and energy use.

## Main Files

```text
run.py                    Main experiment entry point
config.py                 Runtime configuration through environment variables
models.py                 Model loading and generation
prompts.py                System prompts and few-shot prompt variant

game/base.py              Base 20 Questions game
game/tool.py              Tool-use version of the game

evaluation/               Win verification and evaluation logic
word_bank/                Secret targets and hints

submit_model_grid.sh      Launches full model grids on Slurm
submit_bulk_seeds.sh      Launches multiple seeds for one model pairing
run_job.sh                Slurm GPU job script

gather_results.py         Parses Slurm logs into CSV results
postprocessing.ipynb      Analysis notebook used for tables and plots

results/                  Aggregated result CSV files
manual_inspection/        Selected logs used for qualitative inspection
```

## Configuration

Most settings are read from environment variables in `config.py`.

Important variables:

```text
MODE                  standard or tool
PROMPT_VARIANT        default or fewshot
GUESSER_MODEL         Hugging Face model id for the guesser
SECRET_MODEL          Hugging Face model id for the keeper
EXPERIMENT_SEED       random seed
```

In the Slurm grid workflow, `MODE`, `GUESSER_MODEL`, and `SECRET_MODEL` are normally set automatically from a model config CSV.

For example:

```csv
label,mode,models
qwen_main,tool,Qwen/Qwen3-1.7B;Qwen/Qwen3-8B
```

Here the second column sets `MODE=tool`.

For few-shot experiments, the important flag is:

```bash
PROMPT_VARIANT=fewshot
```

Without this flag, the default prompt is used, even if the run name contains `fewshot`.

## Install requirements
Before running the experiment, you should install the requirements:

```
pip install -r requirements.txt
```

## Running One Experiment

Example single run:

```bash
MODE=tool \
GUESSER_MODEL="Qwen/Qwen3-1.7B" \
SECRET_MODEL="Qwen/Qwen3-1.7B" \
EXPERIMENT_SEED=1 \
python3 run.py
```

Few-shot version:

```bash
PROMPT_VARIANT=fewshot \
MODE=tool \
GUESSER_MODEL="Qwen/Qwen3-1.7B" \
SECRET_MODEL="Qwen/Qwen3-1.7B" \
EXPERIMENT_SEED=1 \
python3 run.py
```

Most full experiments were run on the Hendrix GPU cluster rather than locally.

## Running a Model Grid on Slurm (if you have the access)

A full model-family grid is launched with:

```bash
bash submit_model_grid.sh model_configs.csv 5 1 5 30 QwenFixed
```

The arguments are:

```text
model_configs.csv   config file
5                   number of seeds
1                   first seed
5                   batch size
30                  polling interval in seconds
QwenFixed           run-name prefix
```

Few-shot grid example:

```bash
PROMPT_VARIANT=fewshot bash submit_model_grid.sh model_configs_qwen_fewshot.csv 5 1 5 30 QwenFewshot
```

## Bash Script Structure

The Slurm scripts are layered like this:

```text
submit_model_grid.sh
        ↓
submit_bulk_seeds.sh
        ↓
run_job.sh
        ↓
run.py
```

Short explanation:

- `submit_model_grid.sh` reads a CSV and creates all guesser/keeper pairings.
- `submit_bulk_seeds.sh` runs several seeds for one guesser/keeper pairing.
- `run_job.sh` requests GPUs, loads modules, checks CUDA, and starts Python.
- `run.py` runs the actual 100-game experiment and prints the evaluation summary.

By default, model-grid runs are submitted in serial wrapper mode, so later configs may appear in Slurm as `PD (Dependency)`. This is expected.

## Checking Completeness

After a Slurm run, check whether each config has successful summaries:

```bash
for d in runs_<TAG>_*; do
  total=$(ls "$d"/slurm-batch*-seed*.out 2>/dev/null | wc -l)
  ok=$(grep -l "=== EVALUATION SUMMARY" "$d"/slurm-batch*-seed*.out 2>/dev/null | wc -l)
  echo "$d -> $ok/$total successful"
done
```

A `3 x 3` grid with 5 seeds should have 45 successful runs. A `4 x 4` grid with 5 seeds should have 80 successful runs.

## Gathering Results

When the submitted jobs are finished, use `gather_results.py` to parse Slurm logs:

```bash
python3 gather_results.py runs_QwenFixed_*/*.out
```

This writes:

```text
results/results.csv
```

Important: `results/results.csv` is overwritten each time `gather_results.py` is run. Always check the model names before copying it to a final file.

Sanity check:


```bash
python3 - <<'PY'
import csv
p = "results/results.csv"
with open(p, newline="") as f:
    rows = list(csv.DictReader(f))
print("rows:", len(rows))
print("guessers:", sorted({r["guesser_model"] for r in rows}))
print("keepers:", sorted({r["secret_model"] for r in rows}))
print("total games:", sum(int(float(r["num_games"])) for r in rows))
PY
```

Then copy it to a named result file, for example:

```bash
cp results/results.csv results/results_qwen_fixed_2026-05-29.csv
```

## Results

Main result files are in:

```text
results/
```

Few-shot result files are in:

```text
results/fewshot/
```

Manual inspection logs are in:

```text
manual_inspection/
```

Files marked with `BAD_` are intentionally kept as warnings from accidental stale copies and should not be used as final results.

## Notes

Some models require Hugging Face access tokens. On the cluster, `run_job.sh` attempts to load a token from the Hugging Face cache.

Energy and carbon measurements are collected with `carbontracker`. These values should be interpreted as relative signals within our setup, not as hardware-independent measurements.
