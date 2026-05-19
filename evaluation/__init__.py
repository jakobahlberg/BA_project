from __future__ import annotations
from typing import Dict, List
import numpy as np

from evaluation.records import GameRecord, EvaluationResult
from evaluation.layer1 import layer1_game_outcome
from evaluation.win_verifier import verify_win


def evaluate_game(
    record: GameRecord,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> EvaluationResult:
    """
    Score a completed game (outcome + win verification) and return an EvaluationResult.

    Args:
        record:           Completed GameRecord from a single round.
        embed_model_name: Sentence-transformers model for win-verification embeddings.

    Returns:
        EvaluationResult with all scores and details populated.
    """
    win, eff, rel = layer1_game_outcome(record)

    wv = verify_win(record, embed_model_name=embed_model_name)

    return EvaluationResult(
        win_score=win,
        efficiency_score=eff,
        secret_reliability_score=rel,
        hints_used=record.hints_used,
        web_searches_used=record.web_searches_used,
        tool_calls_used=record.hints_used + record.web_searches_used,
        verified_win=wv.verified_win,
        win_confidence=wv.win_confidence,
        best_guess_sim=wv.best_guess_sim,
        leaked=wv.leaked,
        false_correct=wv.false_correct,
        secret_keeper_accuracy=wv.secret_keeper_accuracy,
        details={
            "secret": record.secret,
            "turns_used": record.turns_used,
            "max_turns": record.max_turns,
            "was_correct": record.was_correct,
            "num_questions": len(record.questions),
            "num_guesses": len(record.guesses),
            "hints_used": record.hints_used,
        },
    )


def summarise_results(results: List[EvaluationResult]) -> Dict:
    """
    Aggregate EvaluationResults across all rounds into mean scores.

    Args:
        results: List of EvaluationResult, one per round.

    Returns:
        Dict of avg_<field> for every numeric score field,
        plus num_games and num_wins.
    """
    if not results:
        return {}

    fields = [
        "win_score", "efficiency_score", "secret_reliability_score",
        "hints_used", "web_searches_used", "tool_calls_used",
        "best_guess_sim", "secret_keeper_accuracy",
    ]
    summary: Dict = {}
    for f in fields:
        summary[f"avg_{f}"] = float(np.mean([getattr(r, f) for r in results]))
    summary["num_games"]             = len(results)
    summary["num_wins"]              = int(sum(r.win_score for r in results))
    summary["num_verified_wins"]     = int(sum(r.verified_win for r in results))
    summary["num_high_confidence"]   = int(sum(r.win_confidence == "high" for r in results))
    summary["num_medium_confidence"] = int(sum(r.win_confidence == "medium" for r in results))
    summary["num_leaked"]            = int(sum(r.leaked for r in results))
    summary["num_false_correct"]     = int(sum(r.false_correct for r in results))
    summary["verified_layer1_agreement"] = float(
        np.mean([r.verified_win == bool(r.win_score) for r in results])
    )
    return summary
