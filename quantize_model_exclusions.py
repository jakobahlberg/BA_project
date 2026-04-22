"""
quantize_model_exclusions.py
────────────────────────────
Same as quantize_model.py, but keeps sensitive layers in fp16 by passing a
non-empty exclude_layers list to attach_weight_quantizers.

Motivation: at 2-bit and 4-bit, quantizing the token embedding and LM head
tends to dominate the quality drop. Excluding them isolates the cost of
quantizing the attention/MLP blocks.

Matching: osciquant's attach_weight_quantizers performs substring matching
against model.named_modules() keys, so "lm_head" and "embed_tokens" match
the corresponding Qwen3 modules.

NOTE: osciquant uses fake quantization — weights stay as float16 but their
values are rounded to n-bit precision (see osciquant/quantizers.py: the
UniformQuantizer returns `RoundSTE.apply(x / s) * s`, keeping float dtype).
This measures quality degradation, not memory/speed.

Usage:
    python3 quantize_model_exclusions.py --model Qwen/Qwen3-8B --bits 4 \
        --output ./quantized_models/Qwen3-8B-4bit-excl

    python3 quantize_model_exclusions.py --model Qwen/Qwen3-8B --bits 2 \
        --exclude lm_head,embed_tokens \
        --output ./quantized_models/Qwen3-8B-2bit-excl
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from osciquant.quantizers import UniformQuantizer
from osciquant.handler import attach_weight_quantizers, detach_weight_quantizers


# Defaults tuned for Qwen/Qwen3.5-4B (Qwen3_5ForConditionalGeneration).
# Sources:
#   - Kaitchup, "Qwen3.5 Quantization" (AutoRound recipe): keeps lm_head, norms,
#     and embeddings in 16-bit by default; also recommends leaving linear-attention
#     layers in 16-bit (they degrade badly at low bit widths, especially at long
#     context).
#   - Qwen3.5-4B config.json: hybrid attention (28 linear_attention + 4 full),
#     tie_word_embeddings=true (lm_head shares a tensor with embed_tokens).
#
# The vision tower is intentionally NOT excluded: this game is text-only, so those
# weights are never called during inference and their quantization has no effect
# on measured outputs.
#
# These are substring matches against model.named_modules() keys.
DEFAULT_EXCLUDE = [
    "lm_head",
    "embed_tokens",
    "norm",         # all RMSNorm / LayerNorm modules
    "linear_attn",  # Qwen3.5 linear-attention blocks
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--bits",    type=int, default=4, help="Quantization bit width (e.g. 2, 4, 8)")
    parser.add_argument("--output",  required=True, help="Directory to save the quantized model")
    parser.add_argument(
        "--exclude",
        default=",".join(DEFAULT_EXCLUDE),
        help=(
            "Comma-separated substrings matched against named_modules() keys. "
            "Matching layers are left in fp16. "
            f"Default: {','.join(DEFAULT_EXCLUDE)}"
        ),
    )
    args = parser.parse_args()

    exclude_layers = [s.strip() for s in args.exclude.split(",") if s.strip()]

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    print(f"Attaching {args.bits}-bit quantizers (excluding: {exclude_layers}) ...")
    attach_weight_quantizers(
        model=model,
        exclude_layers=exclude_layers,
        quantizer=UniformQuantizer(bit_width=args.bits),
        enabled=True,
    )

    matched = [n for n, _ in model.named_modules() if any(t in n for t in exclude_layers)]
    print(f"  {len(matched)} module(s) kept in fp16:")
    for n in matched:
        print(f"    {n}")

    print("Baking quantized weights ...")
    detach_weight_quantizers(model, leave_parametrized=True)

    print(f"Saving to {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
