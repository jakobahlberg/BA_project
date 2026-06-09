from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from evaluation.records import GameRecord


_CROSS_ENCODER: Optional[CrossEncoder] = None


def _get_cross_encoder(model_name: str = "cross-encoder/stsb-roberta-large") -> CrossEncoder:
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        print(f"[Evaluator] Loading cross-encoder: {model_name}")
        _CROSS_ENCODER = CrossEncoder(model_name)
    return _CROSS_ENCODER

_PRIMARY_THRESHOLD = 0.50   # cross-encoder sim → verified win (+ structural)
_LEAK_THRESHOLD    = 0.50   # cross-encoder sim → leaked secret in question response


def _keeper_said_correct(raw_response: str) -> bool:
    """True only when the keeper response starts with CORRECT."""
    normalized = raw_response.upper()
    if re.match(r"^\W*NOT\s+CORRECT\b", normalized):
        return False
    return bool(re.match(r"^\W*CORRECT\b", normalized))

@dataclass
class WinVerificationResult:
    """Output of the win verification pipeline for one game."""

    verified_win:    bool
    win_confidence:  str        # "high" | "unverified"
    best_guess_sim:  float      # highest cross-encoder sim across all guesses
    leaked:          bool       # keeper leaked the secret in a question response
    false_correct:   bool       # keeper said CORRECT, our verification disagreed


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _required_token(secret: str) -> str:
    """
    Pick the longest content word of the secret as the required token.
    On length ties, prefer the later word.
    """
    words = _normalize(secret).split()
    if not words:
        return ""
    max_len = max(len(w) for w in words)
    candidates = [w for w in words if len(w) == max_len]
    return candidates[-1]


def _structural_pass(secret: str, guess: str) -> bool:
    """The required token of secret must appear as a token in normalised guess."""
    req = _required_token(secret)
    if not req:
        return False
    return req in _normalize(guess).split()


def _verify_primary(
    guesses: List[str],
    secret: str,
    cross_encoder: CrossEncoder,
    threshold: float = _PRIMARY_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Run the keeper-independent verification on every guess in the game.

    For each guess:
      verified = structural_pass(secret, guess) AND cross_encoder_sim >= threshold

    Returns:
      (verified_win, best_sim_across_all_guesses)
    """
    if not guesses:
        return False, 0.0

    pairs = [(secret, g) for g in guesses]
    sims = cross_encoder.predict(pairs)
    best_sim = float(np.max(sims))

    for guess, sim in zip(guesses, sims):
        if float(sim) >= threshold and _structural_pass(secret, guess):
            return True, best_sim
    return False, best_sim


def _leak_detection(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
    cross_encoder: CrossEncoder,
    threshold: float = _LEAK_THRESHOLD,
) -> bool:
    """
    Did any YES/NO response to a question leak the secret?

    Substring match first (cheap, exact); cross-encoder fallback for paraphrased
    leaks.
    """
    raw_responses = [raw for action, _, raw in turn_log if action == "question"]
    if not raw_responses:
        return False

    norm_secret = _normalize(secret)
    for raw in raw_responses:
        if norm_secret in _normalize(raw):
            return True

    pairs = [(secret, raw) for raw in raw_responses]
    sims = cross_encoder.predict(pairs)
    return bool(np.any(np.asarray(sims) >= threshold))


def _detect_false_correct(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
    cross_encoder: CrossEncoder,
    threshold: float = _PRIMARY_THRESHOLD,
) -> bool:
    """Flag if the keeper said CORRECT on a guess that does NOT pass our verification."""
    for action, content, raw_response in turn_log:
        if action == "guess" and _keeper_said_correct(raw_response):
            if (_structural_pass(secret, content)
                    and float(cross_encoder.predict([(secret, content)])[0]) >= threshold):
                return False
            return True
    return False


def verify_win(
    record: GameRecord,
    cross_encoder_model: str = "cross-encoder/stsb-roberta-large",
) -> WinVerificationResult:
    """
    Run the win verification pipeline on a completed game.

    Primary verification (structural + cross-encoder) runs on every guess and
    ignores the keeper's CORRECT/WRONG response. The cross-encoder is cached
    after first load.
    """
    cross_encoder = _get_cross_encoder(cross_encoder_model)

    verified_win, best_sim = _verify_primary(record.guesses, record.secret, cross_encoder)
    leaked        = _leak_detection(record.turn_log, record.secret, cross_encoder)
    false_correct = _detect_false_correct(record.turn_log, record.secret, cross_encoder)

    return WinVerificationResult(
        verified_win=verified_win,
        win_confidence="high" if verified_win else "unverified",
        best_guess_sim=best_sim,
        leaked=leaked,
        false_correct=false_correct,
    )
