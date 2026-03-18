"""
models.py
─────────
Model loading and text generation utilities.

All model I/O lives here so game logic never imports from transformers directly.
Swap out generate_answer() here to change generation behaviour globally.
"""

from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_name: str) -> tuple:
    """
    Load a causal LM and its tokenizer from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen3-8B").

    Returns:
        (model, tokenizer) tuple ready for generate_answer().
    """
    print(f"[Models] Loading: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    print(f"[Models] Loaded: {model_name}")
    return model, tokenizer


def generate_answer(
    messages: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int = 100,
) -> str:
    """
    Apply the chat template, run greedy/sampled generation, and return the
    decoded response string.

    The assistant reply is appended to `messages` in-place so the caller's
    conversation history stays up to date.

    Args:
        messages:       Chat history in OpenAI format (list of role/content dicts).
        model:          Loaded causal LM.
        tokenizer:      Matching tokenizer.
        max_new_tokens: Maximum tokens to generate per call.

    Returns:
        Decoded response string (stripped, special tokens removed).
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.0,
    )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": answer})
    return answer
