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
    True for Gemma variants that need strict user/assistant alternation cleanup.

    Gemma 2 and Gemma 3 reject system turns / repeated user turns in their
    chat template, so we merge system prompts into the first user message and
    collapse consecutive user messages before rendering.
    """
    name = _name_or_path(tokenizer_or_processor).lower()
    if "gemma" not in name:
        return False
    # Gemma 4 currently uses the processor path and should keep its native template.
    if re.search(r"gemma[-_]?4", name):
        return False
    return True


def _is_gemma_image_text_model(model_name: str) -> bool:
    """
    True for Gemma variants whose canonical HF loader is processor based.

    Gemma 3 4B+ and Gemma 4 E2B/E4B are image-text/any-to-any models. We only
    use their text interface, but they still need AutoProcessor plus
    AutoModelForImageTextToText.
    """
    normalized = model_name.lower()
    return bool(
        re.search(r"gemma-?4-?e[24]b", normalized)
        or re.search(r"gemma-?3-?(4b|12b|27b)", normalized)
    )


def _is_ministral3_model(model_name: str) -> bool:
    """
    True for Mistral3 variants (Ministral-3-*) whose HF loader is processor-based.

    Ministral-3-3B and Ministral-3-8B use Mistral3ForConditionalGeneration with a
    PixtralProcessor, so they need AutoProcessor + AutoModelForImageTextToText
    exactly like Gemma 3/4 multimodal models.
    """
    return "ministral" in model_name.lower()


def _is_ministral(tokenizer_or_processor) -> bool:
    return "ministral" in _name_or_path(tokenizer_or_processor).lower()


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

    Ministral 3 supports an optional system message, but its template requires
    the remaining conversation to alternate user/assistant. The game can append
    consecutive user turns (for example result + reminder), so merge repeated
    roles before rendering.
    """
    if _is_hunyuan(tokenizer_or_processor):
        prepared = []
        for msg in messages:
            if msg["role"] == "user" and not msg["content"].lstrip().startswith("/no_think"):
                prepared.append({"role": "user", "content": "/no_think " + msg["content"]})
            else:
                prepared.append(msg)
        return prepared

    if _is_ministral(tokenizer_or_processor):
        prepared: list[dict] = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if prepared and prepared[-1]["role"] == role and role != "system":
                prepared[-1] = {"role": role, "content": prepared[-1]["content"] + "\n\n" + content}
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


def _processor_messages(messages: list[dict]) -> list[dict]:
    """
    Convert plain string chat messages to the content-list format expected by
    Gemma 3/4 processors.
    """
    converted: list[dict] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": msg.get("role", "user"), "content": content})
    return converted


def _apply_chat_template(messages: list[dict], tokenizer_or_processor) -> str:
    """Render messages with either a tokenizer chat template or processor chat template."""
    template_owner = tokenizer_or_processor
    text_tokenizer = _text_tokenizer(tokenizer_or_processor)
    template_messages = messages

    if not hasattr(template_owner, "apply_chat_template"):
        template_owner = text_tokenizer
    elif hasattr(tokenizer_or_processor, "tokenizer"):
        template_messages = _processor_messages(messages)

    if hasattr(template_owner, "apply_chat_template"):
        try:
            return template_owner.apply_chat_template(
                template_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return template_owner.apply_chat_template(
                template_messages,
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


def _build_inputs(messages: list[dict], model, tokenizer_or_processor):
    """
    Build model inputs using the native processor path when available.

    Gemma 3/4 image-text models should tokenize through
    processor.apply_chat_template(..., tokenize=True). Rendering to a string
    and then calling the processor again can produce bad tokenization for these
    checkpoints.
    """
    if hasattr(tokenizer_or_processor, "tokenizer") and hasattr(tokenizer_or_processor, "apply_chat_template"):
        template_messages = _processor_messages(messages)
        try:
            return tokenizer_or_processor.apply_chat_template(
                template_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(model.device)
        except TypeError:
            return tokenizer_or_processor.apply_chat_template(
                template_messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

    if hasattr(tokenizer_or_processor, "apply_chat_template"):
        try:
            return tokenizer_or_processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
                enable_thinking=False,
            ).to(model.device)
        except TypeError:
            return tokenizer_or_processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)

    text = _apply_chat_template(messages, tokenizer_or_processor)
    return _tokenize_text(text, model, tokenizer_or_processor)


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

    # Multimodal / processor-based models: use AutoProcessor + AutoModelForImageTextToText.
    # Covers Gemma 3 4B+, Gemma 4 E2B/E4B, and Ministral-3 (PixtralProcessor).
    # For text-only inference the processor surface (apply_chat_template / decode)
    # is identical to AutoTokenizer, so generate_answer needs no changes.
    if _is_gemma_image_text_model(model_name) or _is_ministral3_model(model_name):
        if AutoProcessor is None or AutoModelForImageTextToText is None:
            raise RuntimeError(
                "This model requires a recent Transformers version with "
                "AutoProcessor and AutoModelForImageTextToText support."
            )
        processor_kwargs = {"fix_mistral_regex": True} if _is_ministral3_model(model_name) else {}
        try:
            processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        except TypeError:
            processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        print(f"[Models] Loaded (multimodal): {model_name}")
        return model, processor

    tokenizer_kwargs = {"fix_mistral_regex": True} if "mistral" in model_name.lower() else {}
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    except TypeError:
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
    inputs = _build_inputs(prepared, model, tokenizer)

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
