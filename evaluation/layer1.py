from __future__ import annotations
import re
from typing import Tuple
from evaluation.records import GameRecord

_VALID_SECRET_RESPONSE_RE = re.compile(r"^\W*(YES|NO|CORRECT|WRONG|INCORRECT)\b", re.IGNORECASE)

def layer1_game_outcome(record: GameRecord) -> Tuple[float, float, float]:
    """
    Compute Layer 1 scores from a completed GameRecord.

    Scoring:
        win_score:               1.0 if the guesser won, else 0.0
        efficiency_score:        (max_turns - turns_used) / max_turns (only > 0 if won)
        secret_reliability_score: Fraction of secret responses that were valid
                                  one-word answers (YES/NO/CORRECT/WRONG)

    Args:
        record: Completed GameRecord from a single round.

    Returns:
        (win_score, efficiency_score, secret_reliability_score)
    """
    win_score = 1.0 if record.was_correct else 0.0

    MAX_TURNS = 20
    if record.was_correct:
        efficiency_score = (MAX_TURNS - record.turns_used) / MAX_TURNS
    else:
        efficiency_score = 0.0

    if record.secret_raw_responses:
        valid = sum(
            1 for r in record.secret_raw_responses
            if _VALID_SECRET_RESPONSE_RE.search(r)
        )
        secret_reliability_score = valid / len(record.secret_raw_responses)
    else:
        secret_reliability_score = 1.0

    return win_score, efficiency_score, secret_reliability_score
