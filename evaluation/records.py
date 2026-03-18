"""
evaluation/records.py
─────────────────────
Dataclasses for storing game data and evaluation results.
"""

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
    hints_used: int = 0            # Number of hints consumed (hint mode only)


@dataclass
class EvaluationResult:
    """Scores from all three evaluation layers for a single game."""

    # --- Layer 1: Game Outcome ---
    win_score: float
    efficiency_score: float
    secret_reliability_score: float
    layer1_score: float

    # --- Layer 2: Question Quality ---
    semantic_relevance_score: float
    canonical_coverage_score: float
    information_gain_score: float
    layer2_score: float

    # --- Layer 3: LLM Judge ---
    llm_judge_strategy: float
    llm_judge_question_quality: float
    llm_judge_logical_consistency: float
    llm_judge_secret_accuracy: float
    layer3_score: float

    details: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        div_trace = self.details.get("diversity_trace", [])
        div_str = "  " + "  ".join(
            f"Q{i+1}:{v:.2f}" for i, v in enumerate(div_trace)
        ) if div_trace else "  (none)"

        feedbacks = self.details.get("judge_feedbacks", {})

        lines = [
            "=" * 55,
            "EVALUATION RESULTS",
            "=" * 55,
            f"  Secret          : {self.details.get('secret', '?')}",
            f"  Won             : {bool(self.win_score)}",
            f"  Turns used      : {self.details.get('turns_used', '?')} / {self.details.get('max_turns', 20)}",
            "",
            "LAYER 1 — Game Outcome",
            f"  Win                   : {self.win_score:.3f}",
            f"  Efficiency            : {self.efficiency_score:.3f}",
            f"  Secret reliability    : {self.secret_reliability_score:.3f}",
            f"  Layer 1 score         : {self.layer1_score:.3f}",
            "",
            "LAYER 2 — Question Quality",
            f"  Semantic relevance    : {self.semantic_relevance_score:.3f}",
            f"  Canonical coverage    : {self.canonical_coverage_score:.3f}",
            f"  Information gain      : {self.information_gain_score:.3f}",
            f"  Layer 2 score         : {self.layer2_score:.3f}",
            "",
            "  Diversity per question (higher = more novel than prior questions):",
            div_str,
            "",
            "LAYER 3 — Prometheus Judge",
            f"  Strategy              : {self.llm_judge_strategy:.3f}",
        ]
        if "strategy" in feedbacks:
            lines.append(f"    → {feedbacks['strategy']}")
        lines += [f"  Question quality      : {self.llm_judge_question_quality:.3f}"]
        if "question_quality" in feedbacks:
            lines.append(f"    → {feedbacks['question_quality']}")
        lines += [f"  Logical consistency   : {self.llm_judge_logical_consistency:.3f}"]
        if "logical_consistency" in feedbacks:
            lines.append(f"    → {feedbacks['logical_consistency']}")
        lines += [f"  Secret accuracy       : {self.llm_judge_secret_accuracy:.3f}"]
        if "secret_accuracy" in feedbacks:
            lines.append(f"    → {feedbacks['secret_accuracy']}")
        lines += [
            f"  Layer 3 score         : {self.layer3_score:.3f}",
            "=" * 55,
        ]
        return "\n".join(lines)
