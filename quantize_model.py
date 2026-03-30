"""
quantize_model.py
─────────────────
Load a HuggingFace model, apply Post-Training Quantization (PTQ) via osciquant,
bake the quantized weights in permanently, and save to disk.

NOTE: osciquant uses fake quantization — weights stay as float16 but their values
are rounded to n-bit precision. This measures quality degradation, not memory/speed.

Usage:
    python3 quantize_model.py --model Qwen/Qwen3-8B --bits 4 --output ./quantized_models/Qwen3-8B-4bit
    python3 quantize_model.py --model Qwen/Qwen3-8B --bits 8 --output ./quantized_models/Qwen3-8B-8bit
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from osciquant.quantizers import UniformQuantizer
from osciquant.handler import attach_weight_quantizers, detach_weight_quantizers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  required=True, help="HuggingFace model ID or local path")
    parser.add_argument("--bits",   type=int, default=4, help="Quantization bit width (e.g. 2, 4, 8)")
    parser.add_argument("--output", required=True, help="Directory to save the quantized model")
    args = parser.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    print(f"Attaching {args.bits}-bit quantizers ...")
    attach_weight_quantizers(
        model=model,
        exclude_layers=[],
        quantizer=UniformQuantizer(bit_width=args.bits),
        enabled=True,
    )

    print("Baking quantized weights ...")
    detach_weight_quantizers(model, leave_parametrized=True)

    print(f"Saving to {args.output} ...")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()