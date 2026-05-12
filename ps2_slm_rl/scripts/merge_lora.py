"""
scripts/merge_lora.py

Merge a LoRA adapter into the base model weights for clean inference.
The merged model has no adapter overhead and can be loaded with
AutoModelForCausalLM.from_pretrained() directly.

Run:
    python scripts/merge_lora.py \
        --base-model microsoft/Phi-3-mini-4k-instruct \
        --lora-path checkpoints/curriculum_grpo \
        --output-path final_model \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.loader import merge_and_save

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Merge LoRA into base model")
    parser.add_argument("--base-model", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--output-path", default="final_model")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Merging %s + %s → %s", args.base_model, args.lora_path, args.output_path)
    merge_and_save(
        base_model_id=args.base_model,
        lora_checkpoint_path=args.lora_path,
        output_path=args.output_path,
        torch_dtype=args.dtype,
    )
    logger.info("Done. Load with: AutoModelForCausalLM.from_pretrained('%s')", args.output_path)


if __name__ == "__main__":
    main()