import os

# ─── HuggingFace ────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HUGGINGFACE_HUB_TOKEN"] = HF_TOKEN

# ─── Game mode ──────────────────────────────────────────────────────────────
# Options: "standard" | "tool"
MODE = os.environ.get("MODE", "tool")

# ─── Prompt variant ─────────────────────────────────────────────────────────
# Options: "default" | "fewshot"
# The few-shot variant only changes the guesser system prompt. It does not
# change game mechanics, parsing, tools, evaluation, or the secret keeper.
PROMPT_VARIANT = os.environ.get("PROMPT_VARIANT", "default").strip().lower()

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
