"""
game/__init__.py
────────────────
Game mode classes.

  BaseGame  — core loop shared by all modes (standard)
  ToolGame  — extends BaseGame with USE_HINT and WEB_SEARCH actions
"""

from game.base import BaseGame
from game.tool import ToolGame
