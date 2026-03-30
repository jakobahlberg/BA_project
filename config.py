"""
config.py
─────────
Central configuration for all game modes and experiments.

Change MODE to switch between game variants.
All other settings (models, limits, evaluation) are controlled here.
"""

import os

# ─── HuggingFace ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ─── Game mode ──────────────────────────────────────────────────────────────
# Options: "standard" | "tool"
MODE = "tool"

# ─── Models ─────────────────────────────────────────────────────────────────
GUESSER_MODEL = "Qwen/Qwen3.5-4B-Base"
SECRET_MODEL  = "Qwen/Qwen3.5-4B-Base"
JUDGE_MODEL   = "Qwen/Qwen3-8B"

# ─── Game settings ──────────────────────────────────────────────────────────
MAX_TURNS = 20
MAX_HINTS = 5   # only used in tool mode
MAX_WEB_SEARCHES = 5   # only used in tool mode

# ─── Evaluation ─────────────────────────────────────────────────────────────
RUN_JUDGE    = True
DATASET_PATH = "dataset.json"

# ─── Carbon tracking ────────────────────────────────────────────────────────
CARBON_LOG_DIR = "carbon_logs"
