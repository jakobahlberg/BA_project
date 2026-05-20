from __future__ import annotations

import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _is_gemma(tokenizer: AutoTokenizer) -> bool:
    """
    True only for Gemma 1–3, which lack a system role in their chat template.

    Gemma 4 (incl. E2B / E4B) has native system-role support and matches
    Qwen3's chat-template conventions, so it should NOT take the legacy
    merge-system-into-first-user path in _prepare_messages.
    """
    name = tokenizer.name_or_path.lower()
    if "gemma" not in name:
        return False
    # Exclude Gemma 4 variants: "gemma-4", "gemma_4", "gemma4", "gemma-4-e2b", etc.
    if re.search(r"gemma[-_]?4", name):
        return False
    return True


def _is_llama_model_name(model_name: str) -> bool:
    return "llama" in model_name.lower()


def _prepare_messages(messages: list[dict], tokenizer: AutoTokenizer) -> list[dict]:
    """
    Normalize a chat message list for models that don't support a system role.

    Gemma requires strict user/assistant alternation and has no system turn.
    We merge any leading system message into the first user message so the
    conversation starts with a user turn as Gemma expects.
    """
    if not _is_gemma(tokenizer):
        return messages

    prepared: list[dict] = []
    pending_system: str | None = None

    for msg in messages:
        if msg["role"] == "system":
            pending_system = msg["content"]
        elif msg["role"] == "user":
            content = msg["content"]
            if pending_system is not None:
                content = f"{pending_system}\n\n{content}"
                pending_system = None
            if prepared and prepared[-1]["role"] == "user":
                # Merge consecutive user turns — Gemma requires strict alternation
                prepared[-1] = {"role": "user", "content": prepared[-1]["content"] + "\n\n" + content}
            else:
                prepared.append({"role": "user", "content": content})
        else:
            prepared.append(msg)

    return prepared


def _format_plain_chat(messages: list[dict]) -> str:
    """
    Fallback chat formatting for base models without tokenizer chat templates.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


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
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.float16,
    }
    # Mixed GPU fleets can trigger CUDA kernel-image errors with certain fast attention paths.
    # For Llama-family models, prefer eager attention for broader compatibility.
    if _is_llama_model_name(model_name):
        model_kwargs["attn_implementation"] = "eager"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        # Backward-compatible fallback if a transformers version does not accept
        # one of the optional kwargs (e.g., attn_implementation).
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
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
    prepared = _prepare_messages(messages, tokenizer)
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            prepared,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        text = _format_plain_chat(prepared)

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
