from dataclasses import dataclass


@dataclass
class SecretEntry:
    """
    A single secret for one round of 20 Questions.

    Attributes:
        label:         Human-readable name of the secret.
        category:      One of "animal", "food", "object" — used for per-category stats.
        system_prompt: Full system prompt for the secret keeper model.
    """
    label: str
    category: str
    system_prompt: str
