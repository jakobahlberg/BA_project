from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluation.records import GameRecord

_JUDGE_MODEL: Optional[AutoModelForCausalLM] = None
_JUDGE_TOKENIZER: Optional[AutoTokenizer] = None

_JUDGE_PROMPT_TEMPLATE = """You are verifying whether a 20 Questions game was won correctly.

SECRET: {secret}

GUESSES MADE (in order):
{guesses}

Look through each guess. A guess counts as correct if it refers to the same thing as the secret — ignore minor wording differences (e.g. "a dog" matches "dog", "Eiffel Tower" matches "the Eiffel Tower"). The secret keeper may have incorrectly accepted or rejected guesses, so base your verdict solely on whether any guess actually names or clearly refers to the secret.

Did any guess actually match the secret?

Reply in EXACTLY this format, nothing else:
JUDGE_WON: <YES or NO>"""


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



def layer3_llm_judge(
    record: GameRecord,
    judge_model_name: str = "Qwen/Qwen3-8B",
    max_new_tokens: int = 16,
) -> Tuple[float, float, float, float, float, float, float, bool, Dict[str, str]]:
    """
    Determine whether the guesser actually won by checking each guess against
    the real secret using an LLM, bypassing the unreliable secret_keeper signal.

    Returns a tuple compatible with the existing interface; all dimension scores
    are 0.0 since only the win verdict is evaluated here.

    Args:
        record:           Completed GameRecord to evaluate.
        judge_model_name: HuggingFace model to use if not already loaded.
        max_new_tokens:   Token budget for the judge response (small — YES/NO only).

    Returns:
        (strategy, question_quality, logical_consistency, secret_accuracy,
         guesser_format, layer3_score, judge_won, feedbacks_dict)
        All dimension scores are 0.0. judge_won is the verified win verdict.
    """
    global _JUDGE_MODEL, _JUDGE_TOKENIZER

    if _JUDGE_MODEL is None:
        load_judge_model(judge_model_name)

    if not record.guesses:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, {}

    guesses_str = "\n".join(f"  {i+1}. {g}" for i, g in enumerate(record.guesses))

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        secret=record.secret,
        guesses=guesses_str,
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

    judge_won_match = re.search(r"JUDGE_WON:\s*(YES|NO)", raw, re.MULTILINE | re.IGNORECASE)
    if judge_won_match:
        judge_won = judge_won_match.group(1).upper() == "YES"
    else:
        print(f"  [Warning] Could not parse JUDGE_WON from: {raw!r}, defaulting to False")
        judge_won = False

    return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, judge_won, {}
