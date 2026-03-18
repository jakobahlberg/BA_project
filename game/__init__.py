"""
game/__init__.py
────────────────
Game mode classes.

  BaseGame  — core loop shared by all modes (standard, easy)
  HintGame  — extends BaseGame with USE_HINT action and PDF hint delivery
"""

from game.base import BaseGame
from game.hint import HintGame
