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
MODE = os.environ.get("MODE", "tool")

# ─── Models ─────────────────────────────────────────────────────────────────
GUESSER_MODEL = os.environ.get("GUESSER_MODEL", "Qwen/Qwen3.5-4B-Base")
SECRET_MODEL  = os.environ.get("SECRET_MODEL", "Qwen/Qwen3.5-4B-Base")

# ─── Game settings ──────────────────────────────────────────────────────────
MAX_TURNS = 20
MAX_HINTS = 5   # only used in tool mode
MAX_WEB_SEARCHES = 5   # only used in tool mode
MAX_ACTIONS_PER_ROUND = 35   # hard ceiling on guesser actions per round (tool mode); also the CarbonTracker epoch budget

# ─── Carbon tracking ────────────────────────────────────────────────────────
CARBON_LOG_DIR = os.environ.get("CARBON_LOG_DIR", "carbon_logs")
