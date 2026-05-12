"""
scripts/train_sft.py

SFT warm-up training script.
Run: python scripts/train_sft.py --config configs/sft.yaml

HALLUCINATION NOTE:
  - SFTConfig field 'max_length' (not 'max_seq_length') verified in trl 1.4.0.
  - SFTConfig field 'dataset_text_field' verified in trl 1.4.0.
  - gradient_checkpointing_kwargs passed via SFTConfig directly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import yaml
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import build_sft_dataset
from src.model.loader import load_base_model, load_tokenizer, make_lora_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SFT warm-up for Phi-3-Mini")
    parser.add_argument("--config", default="configs/sft.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load data and model but don't train (sanity check)")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # --- W&B init ---
    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["training"]["run_name"],
        config=cfg,
    )

    # --- Data ---
    logger.info("Loading datasets...")
    train_ds, val_ds = build_sft_dataset(
        gsm8k_fraction=cfg["data"]["gsm8k_fraction"],
        aqua_fraction=cfg["data"]["aqua_fraction"],
        val_size=cfg["data"]["val_size"],
    )
    logger.info("Train: %d examples, Val: %d examples", len(train_ds), len(val_ds))

    if args.dry_run:
        logger.info("Dry run: printing 2 training examples then exiting.")
        for i in range(min(2, len(train_ds))):
            print(f"\n--- Example {i} ---")
            print(train_ds[i]["text"][:500])
        return

    # --- Model + LoRA ---
    logger.info("Loading model: %s", cfg["model"]["base_id"])
    model = load_base_model(
        model_id=cfg["model"]["base_id"],
        torch_dtype=cfg["model"]["torch_dtype"],
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    tok = load_tokenizer(cfg["model"]["base_id"])

    lora_cfg = make_lora_config(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )

    # --- SFTConfig ---
    # Field names verified against trl 1.4.0 SFTConfig
    t = cfg["training"]
    sft_cfg = SFTConfig(
        output_dir=t["output_dir"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        # SFT-specific fields (verified trl 1.4.0)
        dataset_text_field="text",
        max_length=cfg["data"]["max_seq_length"],
        packing=False,  # disable sample packing for simplicity
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tok,
        peft_config=lora_cfg,
    )

    logger.info("Starting SFT training...")
    trainer.train()

    logger.info("Saving final SFT checkpoint to %s", t["output_dir"])
    trainer.save_model(t["output_dir"])
    tok.save_pretrained(t["output_dir"])

    wandb.finish()
    logger.info("SFT training complete.")


if __name__ == "__main__":
    main()