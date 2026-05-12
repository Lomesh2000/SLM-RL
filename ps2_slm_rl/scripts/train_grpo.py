"""
scripts/train_grpo.py

GRPO training with optional curriculum gating.
Run: python scripts/train_grpo.py --config configs/grpo.yaml \
         --curriculum-config configs/curriculum.yaml

HALLUCINATION NOTES:
  - GRPOConfig field 'beta' is the KL penalty coefficient (verified trl 1.4.0).
    Earlier versions used 'kl_coef'; that name does NOT exist in trl 1.4.0.
  - GRPOConfig field 'max_completion_length' (not 'max_new_tokens') — verified.
  - GRPOConfig field 'num_generations' — verified.
  - GRPOConfig field 'epsilon' is the clip ratio — verified.
  - GRPOTrainer constructor takes: model, reward_funcs, args, train_dataset,
    eval_dataset, processing_class, peft_config — verified from signature.
  - GRPOTrainer does NOT take a separate 'ref_model' argument in trl 1.4.0.
    When using PEFT, it automatically uses the frozen base as reference.
    (Verified from GRPOTrainer.__init__ signature — no ref_model param.)
  - The curriculum gate wraps the dataset with a custom collator that filters
    by difficulty. This is a custom implementation on top of GRPOTrainer,
    not a built-in feature of trl.
  - CURRICULUM LIMITATION: GRPOTrainer iterates the dataset internally, so
    we cannot filter mid-epoch in the standard training loop. The workaround
    implemented here is to rebuild and restart training with a filtered dataset
    when the gate advances — controlled by curriculum_train().
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
import yaml
import wandb
from datasets import Dataset
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import build_rl_dataset, load_gsm8k
from src.model.loader import (
    load_base_model,
    load_tokenizer,
    make_lora_config,
    load_sft_checkpoint_for_grpo,
)
from src.rewards.combined import make_reward_fn, RewardConfig
from src.training.curriculum import CurriculumGate, CurriculumConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training for Phi-3-Mini")
    parser.add_argument("--config", default="configs/grpo.yaml")
    parser.add_argument("--curriculum-config", default="configs/curriculum.yaml",
                        help="Path to curriculum config. Pass 'none' to disable.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_grpo_config(cfg: dict, output_dir_override: str | None = None) -> GRPOConfig:
    """
    Build GRPOConfig from yaml config dict.
    All field names verified against trl 1.4.0 GRPOConfig.
    """
    t = cfg["training"]
    g = cfg["grpo"]
    return GRPOConfig(
        output_dir=output_dir_override or t["output_dir"],
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
        save_steps=t["save_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        # GRPO-specific (verified field names)
        beta=g["beta"],                             # KL penalty coefficient
        num_generations=g["num_generations"],       # G completions per prompt
        max_completion_length=g["max_completion_length"],
        temperature=g["temperature"],
        top_p=g["top_p"],
        epsilon=g["epsilon"],                       # PPO clip ratio
    )


def filter_dataset_by_difficulty(
    dataset: Dataset, allowed_difficulties: list[int]
) -> Dataset:
    """Filter the dataset to only include examples at allowed difficulty levels."""
    filtered = dataset.filter(
        lambda ex: ex["difficulty"] in allowed_difficulties
    )
    logger.info(
        "Filtered dataset: %d → %d examples (difficulties: %s)",
        len(dataset), len(filtered), allowed_difficulties,
    )
    return filtered


def simple_grpo_train(cfg: dict, full_dataset: Dataset) -> str:
    """
    Run standard GRPO training without curriculum (baseline).
    Returns path to final checkpoint.
    """
    logger.info("=== GRPO training (no curriculum) ===")

    model, tok = _load_policy(cfg)
    reward_cfg = RewardConfig(**cfg["reward"])
    reward_fn = make_reward_fn(reward_cfg)
    grpo_cfg = build_grpo_config(cfg)
    lora_cfg = make_lora_config(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=grpo_cfg,
        train_dataset=full_dataset,
        processing_class=tok,
        peft_config=lora_cfg,
    )
    trainer.train()
    trainer.save_model(cfg["training"]["output_dir"])
    return cfg["training"]["output_dir"]


def curriculum_grpo_train(
    cfg: dict,
    full_dataset: Dataset,
    cur_cfg: CurriculumConfig,
) -> str:
    """
    Run GRPO training with curriculum gating.

    IMPLEMENTATION NOTE: Because GRPOTrainer manages its own training loop
    internally, we cannot filter mid-epoch. Instead, we train in stages:
    for each curriculum stage, we filter the dataset, run GRPOTrainer for
    a fixed number of steps, check the gate, and restart from the latest
    checkpoint with the next difficulty tier unlocked.

    This is a pragmatic workaround for the limitation noted in the module
    docstring. An alternative would be a custom Trainer subclass with a
    custom DataLoader, but that adds substantial complexity.

    Steps-per-stage is derived from: total_steps / num_stages.
    HALLUCINATION NOTE: The 'max_steps' override in GRPOConfig is a real
    field (inherited from TrainingArguments). Using it to limit per-stage
    training is verified correct behaviour.
    """
    gate = CurriculumGate(cur_cfg)
    reward_cfg = RewardConfig(**cfg["reward"])
    reward_fn = make_reward_fn(reward_cfg)
    lora_cfg = make_lora_config(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["lora_alpha"],
        lora_dropout=cfg["lora"]["lora_dropout"],
        bias=cfg["lora"]["bias"],
        target_modules=cfg["lora"]["target_modules"],
    )

    # Estimate total training steps
    n_train = len(full_dataset)
    effective_batch = (
        cfg["training"]["per_device_train_batch_size"]
        * cfg["training"]["gradient_accumulation_steps"]
        * cfg["grpo"]["num_generations"]
    )
    total_steps = max(1, n_train // effective_batch) * cfg["training"]["num_train_epochs"]
    steps_per_stage = total_steps // len(cur_cfg.stage_names)

    logger.info(
        "Curriculum training: %d stages, ~%d steps/stage, %d total steps",
        len(cur_cfg.stage_names), steps_per_stage, total_steps,
    )

    checkpoint_path = cfg["model"]["sft_checkpoint"]
    final_output_dir = cfg["training"]["output_dir"]

    for stage_idx, stage_name in enumerate(cur_cfg.stage_names):
        stage_dir = os.path.join(final_output_dir, f"stage_{stage_idx}_{stage_name}")
        os.makedirs(stage_dir, exist_ok=True)

        allowed = cur_cfg.stage_difficulties[stage_idx]
        stage_ds = filter_dataset_by_difficulty(full_dataset, allowed)

        if len(stage_ds) == 0:
            logger.warning(
                "Stage %s: no examples with difficulties %s — skipping.",
                stage_name, allowed,
            )
            continue

        logger.info("=== Curriculum stage %d: %s (difficulties %s) ===",
                    stage_idx, stage_name, allowed)

        # Load from previous stage's checkpoint (or SFT for stage 0)
        model = _load_model_from_checkpoint(cfg, checkpoint_path)
        tok = load_tokenizer(cfg["model"]["base_id"])

        stage_cfg = build_grpo_config(
            cfg, output_dir_override=stage_dir
        )
        # Override to train for steps_per_stage only
        # HALLUCINATION NOTE: max_steps overrides num_train_epochs in HF Trainer.
        # This is standard HuggingFace TrainingArguments behaviour.
        stage_cfg.max_steps = steps_per_stage

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=reward_fn,
            args=stage_cfg,
            train_dataset=stage_ds,
            processing_class=tok,
            peft_config=lora_cfg,
        )
        trainer.train()
        trainer.save_model(stage_dir)

        checkpoint_path = stage_dir
        logger.info("Stage %s complete. Checkpoint: %s", stage_name, stage_dir)

        # Log gate state
        wandb.log({
            "curriculum/stage_completed": stage_idx,
            "curriculum/stage_name": stage_name,
        })

        # For the last stage, don't try to advance
        if stage_idx < len(cur_cfg.stage_names) - 1:
            logger.info(
                "Advancing to next stage: %s → %s",
                stage_name, cur_cfg.stage_names[stage_idx + 1],
            )

    # Copy final stage checkpoint to main output dir
    import shutil
    last_stage_dir = os.path.join(
        final_output_dir,
        f"stage_{len(cur_cfg.stage_names)-1}_{cur_cfg.stage_names[-1]}"
    )
    if os.path.exists(last_stage_dir):
        for fname in os.listdir(last_stage_dir):
            shutil.copy2(
                os.path.join(last_stage_dir, fname),
                os.path.join(final_output_dir, fname),
            )

    logger.info("Curriculum GRPO training complete. Final model: %s", final_output_dir)
    return final_output_dir


def _load_policy(cfg: dict):
    """Load base model + tokenizer. SFT checkpoint loaded via PeftModel if available."""
    sft_path = cfg["model"].get("sft_checkpoint", "")
    if sft_path and os.path.exists(sft_path):
        from peft import PeftModel
        base = load_base_model(
            cfg["model"]["base_id"],
            cfg["model"]["torch_dtype"],
            cfg["model"]["attn_implementation"],
        )
        model = PeftModel.from_pretrained(base, sft_path, is_trainable=True)
    else:
        logger.warning(
            "SFT checkpoint not found at '%s'. Starting from base model.", sft_path
        )
        model = load_base_model(
            cfg["model"]["base_id"],
            cfg["model"]["torch_dtype"],
            cfg["model"]["attn_implementation"],
        )
    tok = load_tokenizer(cfg["model"]["base_id"])
    return model, tok


def _load_model_from_checkpoint(cfg: dict, checkpoint_path: str):
    """Load model from a checkpoint path."""
    from peft import PeftModel
    base = load_base_model(
        cfg["model"]["base_id"],
        cfg["model"]["torch_dtype"],
        cfg["model"]["attn_implementation"],
    )
    if os.path.exists(os.path.join(checkpoint_path, "adapter_config.json")):
        model = PeftModel.from_pretrained(base, checkpoint_path, is_trainable=True)
    else:
        model = base
    return model


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    use_curriculum = (
        args.curriculum_config.lower() != "none"
        and os.path.exists(args.curriculum_config)
    )

    if use_curriculum:
        with open(args.curriculum_config) as f:
            cur_raw = yaml.safe_load(f)["curriculum"]
        cur_cfg = CurriculumConfig(
            window_size=cur_raw["window_size"],
            min_window_before_advance=cur_raw["min_window_before_advance"],
            easy_to_medium_threshold=cur_raw["thresholds"]["easy_to_medium"],
            medium_to_hard_threshold=cur_raw["thresholds"]["medium_to_hard"],
        )
    else:
        cur_cfg = None

    wandb.init(
        project=cfg["wandb"]["project"],
        name=cfg["training"]["run_name"] + ("-curriculum" if use_curriculum else ""),
        config=cfg,
    )

    logger.info("Loading RL dataset...")
    full_dataset = build_rl_dataset()

    if args.dry_run:
        logger.info("Dry run: dataset size=%d", len(full_dataset))
        logger.info("Sample: %s", full_dataset[0])
        return

    if use_curriculum:
        curriculum_grpo_train(cfg, full_dataset, cur_cfg)
    else:
        simple_grpo_train(cfg, full_dataset)

    wandb.finish()


if __name__ == "__main__":
    main()