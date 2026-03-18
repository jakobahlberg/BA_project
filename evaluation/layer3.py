from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.records import GameRecord

_JUDGE_MODEL: Optional[AutoModelForCausalLM] = None
_JUDGE_TOKENIZER: Optional[AutoTokenizer] = None

_JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator for 20 Questions games.

SECRET: {secret}
OUTCOME: {outcome} in {turns}/{max_turns} turns

TRANSCRIPT:
{transcript}

Score the game on 4 dimensions. For each, write ONE sentence of reasoning then an integer score 1-10.

STRATEGY: Did the guesser use binary search to efficiently narrow down candidates?
  10 = optimal bisection each turn | 1 = random or repetitive questions with no narrowing

QUESTION_QUALITY: Were questions clear, unambiguous, and non-redundant?
  10 = all crisp yes/no questions each covering new ground | 1 = vague or compound questions

LOGICAL_CONSISTENCY: Did the guesser stay consistent with all prior answers?
  10 = never contradicted a prior answer | 1 = frequently contradicted known facts

SECRET_ACCURACY: Did the secret keeper give factually correct YES/NO responses?
  10 = every answer was factually correct | 1 = multiple wrong or contradictory answers

Reply in EXACTLY this format, nothing else:
STRATEGY_REASON: <one sentence>
STRATEGY: <1-10>
QUESTION_QUALITY_REASON: <one sentence>
QUESTION_QUALITY: <1-10>
LOGICAL_CONSISTENCY_REASON: <one sentence>
LOGICAL_CONSISTENCY: <1-10>
SECRET_ACCURACY_REASON: <one sentence>
SECRET_ACCURACY: <1-10>"""


def load_judge_model(model_name: str) -> None:
    """
    Load the judge LLM and cache it globally.

    Call this once before the first evaluate_game() call to avoid
    reloading the model on every round.

    Args:
        model_name: HuggingFace model identifier for the judge.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER
    print(f"[Evaluator] Loading judge model: {model_name}")
    _JUDGE_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    _JUDGE_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print("[Evaluator] Judge model loaded.")


def _build_transcript(record: GameRecord) -> str:
    """
    Build a clean, numbered transcript string from turn_log.

    Falls back to reconstructing from questions/answers/guesses if
    turn_log is empty (e.g. from older GameRecord formats).
    """
    if not record.turn_log:
        lines = []
        for i, (q, a) in enumerate(zip(record.questions, record.answers), 1):
            lines.append(f"Turn {i}: Q: {q}  →  {a}")
        for g in record.guesses:
            outcome = "CORRECT" if record.was_correct and g == record.final_guess else "WRONG"
            lines.append(f"GUESS: {g}  →  {outcome}")
        return "\n".join(lines) if lines else "(no turns recorded)"

    lines = []
    for i, (action, content, response) in enumerate(record.turn_log, 1):
        if action == "question":
            lines.append(f"Turn {i}: Q: {content}  →  {response}")
        else:
            outcome = "CORRECT" if record.was_correct and content == record.final_guess else "WRONG"
            lines.append(f"Turn {i}: GUESS: {content}  →  {outcome}")
    return "\n".join(lines)


def layer3_llm_judge(
    record: GameRecord,
    judge_model_name: str = "Qwen/Qwen3-8B",
    max_new_tokens: int = 256,
) -> Tuple[float, float, float, float, float, Dict[str, str]]:
    """
    Score a game on 4 quality dimensions using a single LLM call.

    Scores are parsed from the model output and normalised to [0, 1].
    If a score cannot be parsed, it defaults to 0.5 with a warning.

    Args:
        record:           Completed GameRecord to evaluate.
        judge_model_name: HuggingFace model to use if not already loaded.
        max_new_tokens:   Token budget for the judge response.

    Returns:
        (strategy, question_quality, logical_consistency, secret_accuracy,
         layer3_score, feedbacks_dict)
        All scores in [0, 1]. layer3_score is the mean of the four.
        feedbacks_dict maps dimension name → one-sentence reason string.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER

    if _JUDGE_MODEL is None:
        load_judge_model(judge_model_name)

    transcript = _build_transcript(record)
    outcome    = "WON" if record.was_correct else "LOST"

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        secret=record.secret,
        outcome=outcome,
        turns=record.turns_used,
        max_turns=record.max_turns,
        transcript=transcript,
    )

    messages = [
        {"role": "system", "content": "You are a fair and precise game evaluator."},
        {"role": "user",   "content": prompt},
    ]

    text = _JUDGE_TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = _JUDGE_TOKENIZER([text], return_tensors="pt").to(_JUDGE_MODEL.device)

    with torch.no_grad():
        output_ids = _JUDGE_MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_JUDGE_TOKENIZER.eos_token_id,
        )

    generated = output_ids[0][inputs.input_ids.shape[-1]:]
    raw = _JUDGE_TOKENIZER.decode(generated, skip_special_tokens=True).strip()

    dims = ("STRATEGY", "QUESTION_QUALITY", "LOGICAL_CONSISTENCY", "SECRET_ACCURACY")
    scores:    Dict[str, float] = {}
    feedbacks: Dict[str, str]  = {}

    for dim in dims:
        score_match  = re.search(rf"^{dim}:\s*(\d+)", raw, re.MULTILINE)
        reason_match = re.search(rf"^{dim}_REASON:\s*(.+)", raw, re.MULTILINE)

        if score_match:
            raw_score = int(score_match.group(1))
            scores[dim.lower()] = max(0.0, min(1.0, (raw_score - 1) / 9.0))
        else:
            scores[dim.lower()] = 0.5
            print(f"  [Warning] Could not parse score for '{dim}'")

        feedbacks[dim.lower()] = reason_match.group(1).strip() if reason_match else ""

    layer3_score = float(np.mean(list(scores.values())))

    return (
        scores["strategy"],
        scores["question_quality"],
        scores["logical_consistency"],
        scores["secret_accuracy"],
        layer3_score,
        feedbacks,
    )
