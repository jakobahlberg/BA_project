from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from evaluation.records import GameRecord


# ── Embedding model cache (loaded once, reused across all rounds) ─────────────

_EMBED_MODEL: Optional[SentenceTransformer] = None


def _get_embed_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load and cache the sentence embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        print(f"[Evaluator] Loading embedding model: {model_name}")
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


# ── Thresholds ────────────────────────────────────────────────────────────────

_STEP2_THRESHOLD         = 0.85   # guess vs. secret → verified win
_STEP3_LEAK_THRESHOLD    = 0.70   # response vs. secret → leaked
_STEP4_CORRECT_THRESHOLD = 0.70   # guess vs. secret when CORRECT was said
_STEP4_FALSE_THRESHOLD   = 0.30   # if below this when CORRECT said → false win
_STEP5_SUSPICIOUS_GUESS  = 0.85   # guess sim threshold for accuracy check
_STEP5_SUSPICIOUS_QUESTION = 0.85 # question sim threshold for accuracy check


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class WinVerificationResult:
    """Output of the full win verification pipeline for one game."""

    verified_win:            bool
    win_confidence:          str        # "high" | "medium" | "unverified"

    # Step 2
    best_guess_sim:           float     # highest cosine sim across all guesses

    # Step 3 (metric only)
    leaked:                  bool

    # Step 4
    false_correct:           bool       # CORRECT said, but sim < false threshold

    # Step 5 (metric only)
    secret_keeper_accuracy:  float


# ── Sentence stripping ────────────────────────────────────────────────────────

_SENTENCE_PREFIX = re.compile(
    r"^(it is (a|an|the)?|i think (it'?s?|it is) (a|an|the)?|"
    r"the answer is (a|an|the)?|i believe (it is|it'?s?) (a|an|the)?)",
    re.IGNORECASE,
)


def _clean_guess(guess: str) -> str:
    """Strip common sentence prefixes to expose the core noun phrase."""
    return _SENTENCE_PREFIX.sub("", guess).strip().rstrip(".")


# ── Normalisation ─────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Lowercase, strip articles and punctuation, collapse whitespace.

    'The Eiffel Tower' → 'eiffel tower'
    'a dog'            → 'dog'
    """
    text = text.lower().strip()
    # Remove accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    # Strip articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Strip punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Step 1 — Normalised exact match after CORRECT ────────────────────────────

def _step1_normalized_match(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
) -> bool:
    """
    Return True if any guess that received CORRECT also matches the secret
    after normalisation.

    This is the cheapest and most reliable signal: the secret keeper approved
    AND the text is literally the same concept.
    """
    norm_secret = _normalize(secret)
    for action, content, raw_response in turn_log:
        if action == "guess" and "CORRECT" in raw_response.upper():
            norm_guess = _normalize(content)
            if norm_guess == norm_secret:
                return True
            if norm_secret in norm_guess or norm_guess in norm_secret:
                return True
    return False


# ── Step 2 — Embedding similarity on all guesses ─────────────────────────────

def _step2_embedding_guesses(
    guesses: List[str],
    secret: str,
    embed_model,
    threshold: float = _STEP2_THRESHOLD,
) -> Tuple[bool, float]:
    """
    Encode all guesses and the secret, return the best cosine similarity.

    Catches games where the guesser found the right answer but the secret
    keeper incorrectly rejected it.

    Returns:
        (verified, best_sim)
    """
    if not guesses:
        return False, 0.0

    cleaned = [_clean_guess(g) for g in guesses]
    # Deduplicate: use cleaned version only when it differs from the original
    all_guess_texts = guesses + [c for c, g in zip(cleaned, guesses) if c != g]

    texts = all_guess_texts + [secret]
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    guess_embs = embs[:-1]
    secret_emb = embs[-1]

    sims = guess_embs @ secret_emb
    best_sim = float(np.max(sims))

    return best_sim >= threshold, best_sim


# ── Step 3 — Leak detection (metric, not used for verified_win) ───────────────

def _step3_leak_detection(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
    embed_model,
    threshold: float = _STEP3_LEAK_THRESHOLD,
) -> bool:
    """
    Scan all YES/NO responses (not guess responses) for leaks.

    A leak is detected when the secret keeper's response to a plain question
    either directly contains the secret string or has embedding similarity
    ≥ threshold with the secret.

    Returns:
        True if any question response leaked the secret.
    """
    question_turns = [
        (content, raw)
        for action, content, raw in turn_log
        if action == "question"
    ]
    if not question_turns:
        return False

    norm_secret = _normalize(secret)
    raw_responses = [raw for _, raw in question_turns]

    # Batch encode all responses + secret
    texts = raw_responses + [secret]
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    resp_embs = embs[:-1]
    secret_emb = embs[-1]
    sims = resp_embs @ secret_emb

    for idx, (_, raw) in enumerate(question_turns):
        if norm_secret in _normalize(raw):
            return True
        if float(sims[idx]) >= threshold:
            return True
    return False


# ── Step 4 — Verified CORRECT cross-check ────────────────────────────────────

def _step4_verified_correct(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
    embed_model,
    threshold: float = _STEP4_CORRECT_THRESHOLD,
    false_threshold: float = _STEP4_FALSE_THRESHOLD,
) -> Tuple[bool, bool]:
    """
    When the secret keeper said CORRECT, verify by embedding similarity.

    Returns:
        (step4_verified, false_correct)

        step4_verified : True if the guess that got CORRECT has sim ≥ threshold
        false_correct  : True if sim < false_threshold (hallucinated CORRECT)
    """
    for action, content, raw_response in turn_log:
        if action == "guess" and "CORRECT" in raw_response.upper():
            embs = embed_model.encode(
                [content, secret], convert_to_numpy=True, normalize_embeddings=True
            )
            sim = float(embs[0] @ embs[1])
            if sim >= threshold:
                return True, False
            elif sim < false_threshold:
                return False, True
            else:
                # In between: not confident enough either way
                return False, False
    # CORRECT was never said
    return False, False


# ── Step 5 — Secret keeper accuracy metric ───────────────────────────────────

def _step5_secret_keeper_accuracy(
    turn_log: List[Tuple[str, str, str]],
    secret: str,
    embed_model,
    guess_threshold: float    = _STEP5_SUSPICIOUS_GUESS,
    question_threshold: float = _STEP5_SUSPICIOUS_QUESTION,
) -> float:
    """
    Estimate how accurately the secret keeper responded.

    Suspicious response: a guess or question that is very similar to the
    secret (suggesting the answer should be CORRECT/YES) but received
    WRONG/NO instead.

    score = 1 - (suspicious / total_turns)

    Note: this metric is approximate — it flags high-similarity turns where
    we'd *expect* affirmative answers, not definitively incorrect ones.
    """
    if not turn_log:
        return 1.0

    contents = [content for _, content, _ in turn_log]
    texts = contents + [secret]
    embs = embed_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    content_embs = embs[:-1]
    secret_emb = embs[-1]
    sims = content_embs @ secret_emb

    suspicious = 0
    for idx, turn in enumerate(turn_log):
        action, raw = turn[0], turn[2]
        sim = float(sims[idx])
        raw_upper = raw.upper()
        if action == "guess" and sim >= guess_threshold:
            if "WRONG" in raw_upper and "CORRECT" not in raw_upper:
                suspicious += 1
        elif action == "question" and sim >= question_threshold:
            if "NO" in raw_upper and "YES" not in raw_upper:
                suspicious += 1

    return 1.0 - (suspicious / len(turn_log))


# ── Main entry point ──────────────────────────────────────────────────────────

def verify_win(
    record: GameRecord,
    embed_model_name: str = "all-MiniLM-L6-v2",
) -> WinVerificationResult:
    """
    Run the full multi-step win verification pipeline on a completed game.

    The embedding model is cached after first load — no extra model is loaded
    on subsequent rounds.

    Args:
        record:           Completed GameRecord.
        embed_model_name: Sentence-transformers model name.

    Returns:
        WinVerificationResult with all signals and confidence level populated.
    """
    embed_model = _get_embed_model(embed_model_name)

    step1 = _step1_normalized_match(record.turn_log, record.secret)

    step2_verified, best_guess_sim = _step2_embedding_guesses(
        record.guesses, record.secret, embed_model
    )

    leaked = _step3_leak_detection(
        record.turn_log, record.secret, embed_model
    )

    step4_verified, false_correct = _step4_verified_correct(
        record.turn_log, record.secret, embed_model
    )

    secret_keeper_accuracy = _step5_secret_keeper_accuracy(
        record.turn_log, record.secret, embed_model
    )

    # ── Confidence decision ───────────────────────────────────────────────────
    if step1 or step2_verified:
        verified_win = True
        win_confidence = "high"
    elif step4_verified and not false_correct:
        verified_win = True
        win_confidence = "medium"
    else:
        verified_win = False
        win_confidence = "unverified"

    return WinVerificationResult(
        verified_win=verified_win,
        win_confidence=win_confidence,
        best_guess_sim=best_guess_sim,
        leaked=leaked,
        false_correct=false_correct,
        secret_keeper_accuracy=secret_keeper_accuracy,
    )
