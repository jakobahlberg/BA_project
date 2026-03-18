"""
secrets/__init__.py
───────────────────
SecretEntry dataclass used by all secret definition files.

Each game mode has its own file (standard.py, easy.py, hint.py) containing
a SECRETS list of SecretEntry objects.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class SecretEntry:
    """
    A single secret for one round of 20 Questions.

    Attributes:
        label:         Exact name of the secret (must match dataset.json "name" field
                       for information-gain scoring to work).
        category:      One of "animal", "food", "object" — used for per-category stats.
        system_prompt: Full system prompt for the secret keeper model.
        hints:         Ordered list of hints revealed one at a time (hint mode only).
    """
    label: str
    category: str
    system_prompt: str
    hints: List[str] = field(default_factory=list)
