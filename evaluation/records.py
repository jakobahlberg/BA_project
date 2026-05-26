from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class GameRecord:
    """Everything captured during one round of 20 Questions."""

    secret: str
    questions: List[str]           # Yes/no questions asked (in order)
    answers: List[str]             # Normalised YES/NO answers (in order)
    guesses: List[str]             # All guesses made (in order)
    final_guess: str               # Last guess (or "" if none made)
    was_correct: bool              # Whether the guesser won
    turns_used: int                # Total turns taken
    secret_raw_responses: List[str]  # Raw model output for every secret turn
    turn_log: List[Tuple[str, str, str]]  # (action, content, raw_response) per turn
    max_turns: int = 20
    hints_used: int = 0            # Number of hints consumed (tool mode only)
    web_searches_used: int = 0     # Number of web searches used (tool mode only)
    raw_transcript: str = ""       # Raw printed game log from start to end


@dataclass
class EvaluationResult:
    """Game-outcome scores and win verification for a single game."""

    # --- Game Outcome ---
    win_score: float
    efficiency_score: float
    secret_reliability_score: float

    # Tool usage (tool mode only; 0 for standard mode)
    hints_used: int = 0
    web_searches_used: int = 0
    tool_calls_used: int = 0       # hints_used + web_searches_used

    # Win verification (multi-step deterministic pipeline)
    verified_win: bool = False
    win_confidence: str = "unverified"   # "high" | "unverified"
    best_guess_sim: float = 0.0
    leaked: bool = False
    false_correct: bool = False

    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "=" * 55,
            "EVALUATION RESULTS",
            "=" * 55,
            f"  Secret          : {self.details.get('secret', '?')}",
            f"  Won             : {bool(self.win_score)}",
            f"  Turns used      : {self.details.get('turns_used', '?')} / {self.details.get('max_turns', 20)}",
            "",
            "GAME OUTCOME",
            f"  Win                   : {self.win_score:.3f}",
            f"  Efficiency            : {self.efficiency_score:.3f}",
            f"  Secret reliability    : {self.secret_reliability_score:.3f}",
            "",
            "WIN VERIFICATION",
            f"  Verified win          : {self.verified_win} ({self.win_confidence} confidence)",
            f"  Best guess sim        : {self.best_guess_sim:.3f}",
            f"  False CORRECT         : {self.false_correct}",
            f"  Leaked                : {self.leaked}",
            "=" * 55,
        ]
        return "\n".join(lines)
