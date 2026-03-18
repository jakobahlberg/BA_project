# BA-LLM — 20 Questions Benchmark for LLMs

A research framework for evaluating LLM reasoning and question-asking ability through the game of 20 Questions. Two LLMs play against each other — one acts as the **guesser**, the other as the **secret keeper** — and the resulting games are evaluated across three layers of metrics.

Built as a Bachelor's thesis project and designed to run on a SLURM GPU cluster (Hendrix).

---

## How it works

- The **secret keeper** is given a secret word (e.g. "golden retriever") and must answer YES/NO questions honestly.
- The **guesser** tries to identify the secret within 20 turns using yes/no questions and guesses.
- After each game, three evaluation layers score the performance automatically.

There are three game modes:

| Mode | Description |
|---|---|
| `standard` | 9 harder secrets (golden retriever, dove, python, ...) |
| `easy` | 9 simpler secrets (dog, cat, elephant, ...) |
| `hint` | 9 secrets where the guesser can request `USE_HINT` when stuck |

---

## Getting started

### 1. Install dependencies

```bash
pip install transformers accelerate torch sentencepiece carbontracker sentence-transformers fpdf2 pymupdf
```

### 2. Configure the run

Edit `config.py` to set your models and game mode:

```python
MODE = "easy"              # "standard" | "easy" | "hint"

GUESSER_MODEL = "Qwen/Qwen3.5-4B"
SECRET_MODEL  = "Qwen/Qwen3.5-4B"
JUDGE_MODEL   = "Qwen/Qwen3-8B"

MAX_TURNS = 20
RUN_JUDGE = True           # set False to skip the LLM-as-judge layer
```

### 3. Run

```bash
python3 run.py
```

On the Hendrix cluster, submit via SLURM instead:

```bash
sbatch run.sh
```

---

## Evaluation layers

Results are printed after each round and summarised at the end. There are three evaluation layers:

- **Layer 1 — Game outcome**: win/loss, turn efficiency, and secret keeper format reliability.
- **Layer 2 — Question quality**: semantic relevance to the secret, canonical coverage (how many distinct conceptual dimensions were explored), and information gain per question.
- **Layer 3 — LLM-as-judge**: a separate model (`JUDGE_MODEL`) reads the full transcript and scores strategy, question quality, logical consistency, and secret accuracy on a 1–10 scale.

See `eval_metrics.md` for a detailed breakdown of every metric and how scores are computed.

---

## Project structure

```
run.py          — entry point; loads models, runs all rounds, prints summary
config.py       — all settings (mode, models, limits, evaluation flags)
prompts.py      — all system prompts for guesser and secret keeper
models.py       — model loading utilities
dataset.json    — pool of 28 objects with canonical Q&A attributes
game/           — game logic (base game and hint variant)
secrets/        — secret definitions for each mode (easy, standard, hint)
evaluation/     — metric computation across all three layers
outputs/        — saved game transcripts and results
```
