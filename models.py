from __future__ import annotations

import re

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

try:
    from transformers import AutoModelForImageTextToText, AutoProcessor
except ImportError:
    AutoModelForImageTextToText = None
    AutoProcessor = None


def _text_tokenizer(tokenizer_or_processor):
    """Return the inner text tokenizer when given a multimodal processor."""
    return getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)


def _name_or_path(tokenizer_or_processor) -> str:
    text_tokenizer = _text_tokenizer(tokenizer_or_processor)
    return getattr(text_tokenizer, "name_or_path", getattr(tokenizer_or_processor, "name_or_path", ""))


def _is_gemma(tokenizer_or_processor) -> bool:
    """
    True only for Gemma 1–3, which lack a system role in their chat template.

    Gemma 4 (incl. E2B / E4B) has native system-role support and matches
    Qwen3's chat-template conventions, so it should NOT take the legacy
    merge-system-into-first-user path in _prepare_messages.
    """
    name = _name_or_path(tokenizer_or_processor).lower()
    if "gemma" not in name:
        return False
    # Exclude Gemma 4 variants: "gemma-4", "gemma_4", "gemma4", "gemma-4-e2b", etc.
    if re.search(r"gemma[-_]?4", name):
        return False
    return True


def _is_gemma4_nano(model_name: str) -> bool:
    """
    True for the multimodal Gemma 4 nano variants (E2B / E4B).

    These models are MatFormer-style multimodal (text + image + audio) and
    their canonical HF loader is AutoProcessor + AutoModelForImageTextToText.
    For text-only inference the processor still exposes the same surface
    (apply_chat_template, __call__, decode) so the rest of the pipeline
    works unchanged.
    """
    return bool(re.search(r"gemma-?4-?e[24]b", model_name.lower()))


def _is_llama_model_name(model_name: str) -> bool:
    return "llama" in model_name.lower()


def _is_hunyuan(tokenizer_or_processor) -> bool:
    return "hunyuan" in _name_or_path(tokenizer_or_processor).lower()


def _eos_token_id(model, tokenizer_or_processor) -> int | None:
    """Return eos_token_id whether given a tokenizer or a multimodal processor."""
    text_tokenizer = _text_tokenizer(tokenizer_or_processor)
    eos = getattr(text_tokenizer, "eos_token_id", None)
    if eos is not None:
        return eos
    return getattr(model.generation_config, "eos_token_id", None)


def _prepare_messages(messages: list[dict], tokenizer_or_processor) -> list[dict]:
    """
    Normalize a chat message list for models that don't support a system role.

    Gemma requires strict user/assistant alternation and has no system turn.
    We merge any leading system message into the first user message so the
    conversation starts with a user turn as Gemma expects.

    Hunyuan: force fast-thinking. The enable_thinking=False kwarg in
    _apply_chat_template is honored by the 1.8B/4B chat templates but ignored
    by the 7B template (which leaves reasoning on and gets stuck in <think>
    loops). The content-level /no_think control is obeyed by all Hunyuan
    variants, so we prepend it to each user turn. New dicts are built so the
    caller's persistent conversation history stays clean across turns.
    """
    if _is_hunyuan(tokenizer_or_processor):
        prepared = []
        for msg in messages:
            if msg["role"] == "user" and not msg["content"].lstrip().startswith("/no_think"):
                prepared.append({"role": "user", "content": "/no_think " + msg["content"]})
            else:
                prepared.append(msg)
        return prepared

    if not _is_gemma(tokenizer_or_processor):
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


def _apply_chat_template(messages: list[dict], tokenizer_or_processor) -> str:
    """Render messages with either a tokenizer chat template or processor chat template."""
    template_owner = tokenizer_or_processor
    text_tokenizer = _text_tokenizer(tokenizer_or_processor)

    if not hasattr(template_owner, "apply_chat_template"):
        template_owner = text_tokenizer

    if hasattr(template_owner, "apply_chat_template"):
        try:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return template_owner.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

    return _format_plain_chat(messages)


def _tokenize_text(text: str, model, tokenizer_or_processor):
    """Tokenize text with either a tokenizer or a multimodal processor."""
    if hasattr(tokenizer_or_processor, "tokenizer") and callable(tokenizer_or_processor):
        try:
            return tokenizer_or_processor(text=[text], return_tensors="pt").to(model.device)
        except TypeError:
            return tokenizer_or_processor(text=text, return_tensors="pt").to(model.device)

    return tokenizer_or_processor([text], return_tensors="pt").to(model.device)


def load_model(model_name: str) -> tuple:
    """
    Load a causal LM (or multimodal model) and its tokenizer/processor
    from HuggingFace.

    Args:
        model_name: HuggingFace model identifier (e.g. "Qwen/Qwen3-8B").

    Returns:
        (model, tokenizer_or_processor) tuple ready for generate_answer().
        For Gemma 4 nano variants (E2B / E4B) the second element is an
        AutoProcessor; for everything else it is an AutoTokenizer.
    """
    print(f"[Models] Loading: {model_name}")
    model_kwargs = {
        "device_map": "auto",
        "dtype": torch.float16,
    }

    # Gemma 4 nano (E2B / E4B): multimodal — use processor + image-text-to-text class.
    # For text-only use the processor's apply_chat_template / __call__ / decode
    # behave identically to AutoTokenizer, so generate_answer needs no changes.
    if _is_gemma4_nano(model_name):
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError(
                "Gemma 4 requires a recent Transformers version with "
                "AutoProcessor and AutoModelForImageTextToText support."
            )
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        print(f"[Models] Loaded (multimodal): {model_name}")
        return model, processor

    tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    model,
    tokenizer,
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
        tokenizer:      Matching tokenizer or processor.
        max_new_tokens: Maximum tokens to generate per call.

    Returns:
        Decoded response string (stripped, special tokens removed).
    """
    prepared = _prepare_messages(messages, tokenizer)
    text = _apply_chat_template(prepared, tokenizer)

    inputs = _tokenize_text(text, model, tokenizer)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        pad_token_id=_eos_token_id(model, tokenizer),
        repetition_penalty=1.0,
    )

    generated_tokens = outputs[0][inputs.input_ids.shape[-1]:]
    answer = _text_tokenizer(tokenizer).decode(generated_tokens, skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": answer})
    return answer
