from __future__ import annotations
from typing import Dict, List
import numpy as np

from evaluation.records import GameRecord, EvaluationResult
from evaluation.layer1 import layer1_game_outcome
from evaluation.layer2 import layer2_question_quality
from evaluation.layer3 import layer3_llm_judge, load_judge_model


def evaluate_game(
    record: GameRecord,
    dataset_path: str = "dataset.json",
    embed_model_name: str = "all-MiniLM-L6-v2",
    judge_model_name: str = "Qwen/Qwen3-8B",
    run_judge: bool = True,
) -> EvaluationResult:
    """
    Run all three evaluation layers on a completed game and return an EvaluationResult.

    Args:
        record:           Completed GameRecord from a single round.
        dataset_path:     Path to dataset.json (used for Layer 2 IG and coverage).
        embed_model_name: Sentence-transformers model for Layer 2 embeddings.
        judge_model_name: HuggingFace model for the Layer 3 judge.
        run_judge:        Set False to skip Layer 3 (faster, no extra model needed).

    Returns:
        EvaluationResult with all scores and details populated.
    """
    win, eff, rel, l1 = layer1_game_outcome(record)

    sem_rel, coverage, ig, l2, div_trace = layer2_question_quality(
        record, dataset_path=dataset_path, embed_model_name=embed_model_name
    )

    if run_judge:
        strat, qq, lc, sa, l3, feedbacks = layer3_llm_judge(
            record, judge_model_name=judge_model_name
        )
    else:
        strat = qq = lc = sa = l3 = 0.0
        feedbacks = {}

    return EvaluationResult(
        win_score=win,
        efficiency_score=eff,
        secret_reliability_score=rel,
        layer1_score=l1,
        semantic_relevance_score=sem_rel,
        canonical_coverage_score=coverage,
        information_gain_score=ig,
        layer2_score=l2,
        llm_judge_strategy=strat,
        llm_judge_question_quality=qq,
        llm_judge_logical_consistency=lc,
        llm_judge_secret_accuracy=sa,
        layer3_score=l3,
        details={
            "secret": record.secret,
            "turns_used": record.turns_used,
            "max_turns": record.max_turns,
            "was_correct": record.was_correct,
            "num_questions": len(record.questions),
            "num_guesses": len(record.guesses),
            "hints_used": record.hints_used,
            "diversity_trace": div_trace,
            "judge_feedbacks": feedbacks,
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
        "win_score", "efficiency_score", "secret_reliability_score", "layer1_score",
        "semantic_relevance_score", "canonical_coverage_score", "information_gain_score",
        "layer2_score", "llm_judge_strategy", "llm_judge_question_quality",
        "llm_judge_logical_consistency", "llm_judge_secret_accuracy", "layer3_score",
    ]
    summary: Dict = {}
    for f in fields:
        summary[f"avg_{f}"] = float(np.mean([getattr(r, f) for r in results]))
    summary["num_games"] = len(results)
    summary["num_wins"]  = int(sum(r.win_score for r in results))
    return summary
