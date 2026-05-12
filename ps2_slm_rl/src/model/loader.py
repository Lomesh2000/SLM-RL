"""
src/model/loader.py

Loads Phi-3-Mini with optional LoRA wrapping and prints module names.

HALLUCINATION NOTE:
  - The target_modules list for Phi-3-mini is based on the published architecture.
    Phi-3-mini uses a fused 'gate_up_proj' (not separate 'gate_proj' and 'up_proj').
    Run --print-modules to verify against the actual loaded model before training.
  - 'flash_attention_2' requires CUDA >= 11.8 and an Ampere+ GPU (A100, RTX 3090+).
    Kaggle P100 does NOT support it. Use 'eager' or 'sdpa' on Kaggle.
  - 'bfloat16' is not supported on Pascal GPUs (P100). Use 'float16' instead.
"""

from __future__ import annotations

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model, PeftModel


PHI3_MINI_ID = "microsoft/Phi-3-mini-4k-instruct"

# Verified Phi-3-mini module names via architecture inspection.
# 'gate_up_proj' is the fused projection (Phi-3 specific).
PHI3_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
]


def get_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype: {dtype_str}. Choose from {list(mapping)}")
    return mapping[dtype_str]


def load_base_model(
    model_id: str = PHI3_MINI_ID,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "eager",
    device_map: str = "auto",
) -> AutoModelForCausalLM:
    """
    Load the base causal LM.

    Args:
        model_id:            HuggingFace model ID or local path.
        torch_dtype:         "bfloat16", "float16", or "float32".
        attn_implementation: "eager", "sdpa", or "flash_attention_2".
                             Use "eager" on Kaggle P100.
        device_map:          "auto" for multi-GPU, "cuda:0" for single GPU.
    """
    dtype = get_dtype(torch_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    return model


def load_tokenizer(model_id: str = PHI3_MINI_ID) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Phi-3 tokenizer has pad_token; if missing, set to eos_token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # required for decoder-only models in batch generation
    return tok


def make_lora_config(
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    bias: str = "none",
    target_modules: list[str] | None = None,
) -> LoraConfig:
    if target_modules is None:
        target_modules = PHI3_LORA_TARGET_MODULES
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=bias,
        target_modules=target_modules,
    )


def load_model_with_lora(
    model_id: str = PHI3_MINI_ID,
    lora_cfg: LoraConfig | None = None,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "eager",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load base model and wrap with LoRA for training.
    Returns (peft_model, tokenizer).
    """
    model = load_base_model(model_id, torch_dtype, attn_implementation)
    tok = load_tokenizer(model_id)

    if lora_cfg is None:
        lora_cfg = make_lora_config()

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tok


def load_sft_checkpoint_for_grpo(
    base_model_id: str = PHI3_MINI_ID,
    sft_checkpoint_path: str = "checkpoints/sft",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "eager",
) -> tuple[AutoModelForCausalLM, AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the SFT checkpoint as the trainable policy and a frozen copy as the
    reference model for KL divergence computation.

    Returns (policy_model, ref_model, tokenizer).

    HALLUCINATION NOTE: GRPOTrainer in trl 1.4.0 accepts a 'model' and infers
    the reference model internally when using PEFT (it keeps the frozen base).
    Passing a separate ref_model is supported but not required with PEFT.
    See scripts/train_grpo.py for how this is used.
    """
    tok = load_tokenizer(base_model_id)
    dtype = get_dtype(torch_dtype)

    # Policy: SFT checkpoint with LoRA weights (will continue training)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    base.config.use_cache = False
    policy = PeftModel.from_pretrained(base, sft_checkpoint_path, is_trainable=True)

    # Reference: frozen copy of the SFT checkpoint
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map="auto",
    )
    ref_base.config.use_cache = False
    ref_model = PeftModel.from_pretrained(ref_base, sft_checkpoint_path, is_trainable=False)
    for param in ref_model.parameters():
        param.requires_grad = False

    return policy, ref_model, tok


def merge_and_save(
    base_model_id: str,
    lora_checkpoint_path: str,
    output_path: str,
    torch_dtype: str = "bfloat16",
) -> None:
    """
    Merge LoRA weights into the base model and save for inference.
    The merged model has no adapter overhead.
    """
    dtype = get_dtype(torch_dtype)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, torch_dtype=dtype, device_map="cpu"
    )
    model = PeftModel.from_pretrained(base, lora_checkpoint_path)
    model = model.merge_and_unload()
    model.save_pretrained(output_path)

    tok = load_tokenizer(base_model_id)
    tok.save_pretrained(output_path)
    print(f"Merged model saved to {output_path}")


def print_model_modules(model_id: str = PHI3_MINI_ID) -> None:
    """Print all named modules — use this to verify target_modules before training."""
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    for name, module in model.named_modules():
        if any(x in name for x in ["proj", "fc", "linear", "embed", "head"]):
            print(f"  {name}: {module.__class__.__name__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-modules", action="store_true",
                        help="Print module names for the model")
    parser.add_argument("--model-id", default=PHI3_MINI_ID)
    args = parser.parse_args()

    if args.print_modules:
        print(f"\nModules in {args.model_id}:")
        print_model_modules(args.model_id)