"""
train.py — HuggingFace Training Job entrypoint

HF Training Jobs call this file directly: python train.py
All configuration is driven by environment variables set in the HF job UI,
with sensible defaults for the baseline run.

STAGE CONTROL (set via HF job env vars):
  TRAIN_STAGE=baseline     → record Phi-3-Mini zero-shot GSM8K score only
  TRAIN_STAGE=sft          → SFT warm-up (requires baseline done first)
  TRAIN_STAGE=grpo         → GRPO without curriculum
  TRAIN_STAGE=curriculum   → GRPO with curriculum gating (full pipeline)

For the FIRST run, set TRAIN_STAGE=baseline.
This records the baseline score and exits — fast, cheap, confirms the setup works.

REQUIRED ENV VARS (set in HF job secrets/env):
  HF_TOKEN          → your HuggingFace token (for model download + output push)
  WANDB_API_KEY     → your W&B key (optional; training still runs without it)

OPTIONAL ENV VARS:
  TRAIN_STAGE            default: baseline
  OUTPUT_REPO            default: (your-hf-username)/ps2-slm-rl-checkpoints
  BASE_MODEL_ID          default: microsoft/Phi-3-mini-4k-instruct
  EVAL_LIMIT             default: 500   (GSM8K examples to eval on)
  BATCH_SIZE             default: 1
  GRAD_ACCUM             default: 16
  LR                     default: 2e-4  (SFT) / 1e-5 (GRPO)
  NUM_EPOCHS             default: 2     (SFT) / 1 (GRPO)
  LORA_R                 default: 16
  GRPO_BETA              default: 0.05
  GRPO_NUM_GENERATIONS   default: 8
  USE_CURRICULUM         default: 1     (1=yes, 0=no; only used in curriculum stage)
  HF_HUB_ENABLE_HF_TRANSFER  default: 1   (set to 1 for faster model downloads)
"""

from __future__ import annotations

import json
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))


def env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))


def env_bool(key: str, default: bool = True) -> bool:
    val = os.environ.get(key, "")
    if not val:
        return default
    return val.strip().lower() not in ("0", "false", "no")


# ---------------------------------------------------------------------------
# Setup: hf_transfer, wandb, token
# ---------------------------------------------------------------------------

def setup_environment():
    # Enable fast HF model downloads
    if env_bool("HF_HUB_ENABLE_HF_TRANSFER", True):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        try:
            import hf_transfer  # noqa: F401
            logger.info("hf_transfer enabled — faster model downloads")
        except ImportError:
            logger.warning("hf_transfer not installed; falling back to standard download")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    # HF token
    hf_token = env("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        logger.info("HuggingFace login OK")
    else:
        logger.warning("HF_TOKEN not set — model download may fail for gated models")

    # W&B
    wandb_key = env("WANDB_API_KEY")
    if wandb_key:
        import wandb
        wandb.login(key=wandb_key)
        logger.info("W&B login OK")
    else:
        logger.info("WANDB_API_KEY not set — disabling W&B logging")
        os.environ["WANDB_DISABLED"] = "true"

    # CUDA check
    import torch
    logger.info("PyTorch: %s", torch.__version__)
    if torch.cuda.is_available():
        logger.info("GPU: %s | VRAM: %.1f GB | bfloat16: %s",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9,
                    torch.cuda.is_bf16_supported())
    else:
        logger.warning("No GPU detected — training will be extremely slow")


# ---------------------------------------------------------------------------
# Build config dicts from env vars (overrides yaml defaults)
# ---------------------------------------------------------------------------

def build_config() -> dict:
    import torch
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    base_model = env("BASE_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
    stage = env("TRAIN_STAGE", "baseline")

    cfg = {
        "stage": stage,
        "model": {
            "base_id": base_model,
            "attn_implementation": "eager",   # safe default; flash_attn_2 needs explicit install
            "torch_dtype": "bfloat16" if bf16_ok else "float16",
        },
        "lora": {
            "r": env_int("LORA_R", 16),
            "lora_alpha": env_int("LORA_R", 16) * 2,   # alpha = 2 * r
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            # Phi-3-mini module names — run scripts/run_baseline.py --print-modules to verify
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_up_proj", "down_proj",
            ],
        },
        "data": {
            "gsm8k_fraction": 0.80,
            "aqua_fraction": 0.20,
            "val_size": 200,
            "max_seq_length": 1024,
        },
        "training": {
            "output_dir": "/tmp/checkpoints/sft",
            "num_train_epochs": env_int("NUM_EPOCHS", 2),
            "per_device_train_batch_size": env_int("BATCH_SIZE", 1),
            "gradient_accumulation_steps": env_int("GRAD_ACCUM", 16),
            "learning_rate": env_float("LR", 2e-4),
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.05,
            "bf16": bf16_ok,
            "fp16": not bf16_ok,
            "gradient_checkpointing": True,
            "logging_steps": 20,
            "save_strategy": "epoch",
            "eval_strategy": "steps",
            "eval_steps": 200,
            "save_total_limit": 2,
            # Use wandb only if API key is explicitly set — safe to call before setup_environment()
            "report_to": "wandb" if os.environ.get("WANDB_API_KEY") else "none",
            "run_name": f"ps2-{stage}",
        },
        "grpo": {
            "output_dir": "/tmp/checkpoints/grpo",
            "beta": env_float("GRPO_BETA", 0.05),
            "num_generations": env_int("GRPO_NUM_GENERATIONS", 8),
            "max_completion_length": 512,
            "temperature": 0.8,
            "top_p": 0.95,
            "epsilon": 0.2,
        },
        "reward": {
            "outcome_weight": 1.0,
            "process_weight": 0.3,
            "step_weight": 0.1,
            "format_penalty": -0.1,
            "reward_cap": 1.5,
        },
        "eval": {
            "limit": env_int("EVAL_LIMIT", 500),
            "batch_size": env_int("BATCH_SIZE", 4),
            "tasks": ["gsm8k"],  # baseline: GSM8K only; expand in later stages
        },
        "output_repo": env("OUTPUT_REPO", ""),
    }
    return cfg


# ---------------------------------------------------------------------------
# Stage: baseline
# ---------------------------------------------------------------------------

def run_baseline(cfg: dict) -> dict:
    """
    Evaluate Phi-3-Mini zero-shot on GSM8K.
    Records the baseline score to /tmp/results/baseline.json
    and optionally pushes it to the output HF repo.
    Returns the scores dict.
    """
    logger.info("=" * 60)
    logger.info("STAGE: baseline")
    logger.info("Model: %s", cfg["model"]["base_id"])
    logger.info("GSM8K limit: %d examples", cfg["eval"]["limit"])
    logger.info("=" * 60)

    from lm_eval import simple_evaluate

    model_args = (
        f"pretrained={cfg['model']['base_id']},"
        f"trust_remote_code=True,"
        f"dtype={cfg['model']['torch_dtype']}"
    )

    logger.info("Running lm_eval on GSM8K (zero-shot)...")
    results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["gsm8k"],
        num_fewshot=0,
        batch_size=cfg["eval"]["batch_size"],
        limit=cfg["eval"]["limit"],
        log_samples=False,
    )

    gsm8k_raw = results["results"]["gsm8k"]
    logger.info("GSM8K raw result keys: %s", list(gsm8k_raw.keys()))

    # Try all known metric key variants
    score = None
    for key in ["exact_match,flexible-extract", "exact_match,strict-match", "acc,none"]:
        if key in gsm8k_raw:
            score = gsm8k_raw[key]
            logger.info("Using metric key: %s", key)
            break

    if score is None:
        logger.error("Could not find score in result: %s", gsm8k_raw)
        score = 0.0

    scores = {
        "stage": "baseline",
        "model": cfg["model"]["base_id"],
        "gsm8k_zero_shot": round(score, 4),
        "gsm8k_zero_shot_pct": round(score * 100, 2),
        "eval_limit": cfg["eval"]["limit"],
        "raw_keys": gsm8k_raw,
    }

    logger.info("=" * 60)
    logger.info("BASELINE GSM8K: %.2f%%", score * 100)
    logger.info("=" * 60)

    # Save locally
    os.makedirs("/tmp/results", exist_ok=True)
    with open("/tmp/results/baseline.json", "w") as f:
        json.dump(scores, f, indent=2)
    logger.info("Saved: /tmp/results/baseline.json")

    # Push to HF repo if configured
    _push_results_to_hub(cfg, "/tmp/results/baseline.json", "results/baseline.json")

    return scores


# ---------------------------------------------------------------------------
# Stage: SFT
# ---------------------------------------------------------------------------

def run_sft(cfg: dict) -> str:
    """
    Run SFT warm-up. Returns path to saved checkpoint.
    Checkpoint is also pushed to HF hub repo if OUTPUT_REPO is set.
    """
    logger.info("=" * 60)
    logger.info("STAGE: sft")
    logger.info("=" * 60)

    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer
    from peft import LoraConfig, TaskType

    from src.data.dataset import build_sft_dataset
    from src.model.loader import load_base_model, load_tokenizer

    # Data
    logger.info("Loading SFT datasets...")
    train_ds, val_ds = build_sft_dataset(
        gsm8k_fraction=cfg["data"]["gsm8k_fraction"],
        aqua_fraction=cfg["data"]["aqua_fraction"],
        val_size=cfg["data"]["val_size"],
    )

    # Model
    logger.info("Loading model: %s", cfg["model"]["base_id"])
    model = load_base_model(
        model_id=cfg["model"]["base_id"],
        torch_dtype=cfg["model"]["torch_dtype"],
        attn_implementation=cfg["model"]["attn_implementation"],
    )
    tok = load_tokenizer(cfg["model"]["base_id"])

    # LoRA
    lc = cfg["lora"]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lc["r"],
        lora_alpha=lc["lora_alpha"],
        lora_dropout=lc["lora_dropout"],
        bias=lc["bias"],
        target_modules=lc["target_modules"],
    )

    # Training config
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
        fp16=t.get("fp16", False),
        gradient_checkpointing=t["gradient_checkpointing"],
        logging_steps=t["logging_steps"],
        save_strategy=t["save_strategy"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_total_limit=t["save_total_limit"],
        report_to=t["report_to"],
        run_name=t["run_name"],
        dataset_text_field="text",
        max_length=cfg["data"]["max_seq_length"],
        packing=False,
    )

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
    trainer.save_model(t["output_dir"])
    tok.save_pretrained(t["output_dir"])
    logger.info("SFT checkpoint saved: %s", t["output_dir"])

    _push_checkpoint_to_hub(cfg, t["output_dir"], "sft")
    return t["output_dir"]


# ---------------------------------------------------------------------------
# Stage: GRPO / Curriculum GRPO
# ---------------------------------------------------------------------------

def run_grpo(cfg: dict, use_curriculum: bool = False) -> str:
    """
    Run GRPO training (with or without curriculum).
    Returns path to final checkpoint.
    """
    stage_name = "curriculum_grpo" if use_curriculum else "grpo"
    logger.info("=" * 60)
    logger.info("STAGE: %s", stage_name)
    logger.info("=" * 60)

    # Delegate to the existing grpo training script logic
    # Import here to avoid loading heavy deps during baseline
    import yaml as _yaml
    from scripts.train_grpo import simple_grpo_train, curriculum_grpo_train
    from src.data.dataset import build_rl_dataset
    from src.training.curriculum import CurriculumConfig

    # Patch cfg into the format train_grpo.py expects
    grpo_script_cfg = {
        "model": {
            **cfg["model"],
            "sft_checkpoint": cfg["training"]["output_dir"],
        },
        "lora": cfg["lora"],
        "training": {
            **cfg["training"],
            "output_dir": cfg["grpo"]["output_dir"],
            "learning_rate": env_float("LR", 1e-5),
            "num_train_epochs": env_int("NUM_EPOCHS", 1),
            "run_name": f"ps2-{stage_name}",
            "save_strategy": "steps",
            "save_steps": 200,
        },
        "grpo": cfg["grpo"],
        "reward": cfg["reward"],
    }

    full_dataset = build_rl_dataset()

    if use_curriculum:
        cur_cfg = CurriculumConfig()  # uses defaults from curriculum.py
        out_dir = curriculum_grpo_train(grpo_script_cfg, full_dataset, cur_cfg)
    else:
        out_dir = simple_grpo_train(grpo_script_cfg, full_dataset)

    _push_checkpoint_to_hub(cfg, out_dir, stage_name)
    return out_dir


# ---------------------------------------------------------------------------
# HF Hub push helpers
# ---------------------------------------------------------------------------

def _push_results_to_hub(cfg: dict, local_path: str, repo_path: str) -> None:
    """Push a results file to the output HF repo."""
    repo = cfg.get("output_repo", "")
    if not repo:
        logger.info("OUTPUT_REPO not set — skipping hub push for %s", local_path)
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repo_path,
            repo_id=repo,
            repo_type="model",
        )
        logger.info("Pushed %s → %s/%s", local_path, repo, repo_path)
    except Exception as e:
        logger.error("Hub push failed: %s", e)


def _push_checkpoint_to_hub(cfg: dict, local_dir: str, subfolder: str) -> None:
    """Push a checkpoint directory to the output HF repo."""
    repo = cfg.get("output_repo", "")
    if not repo:
        logger.info("OUTPUT_REPO not set — skipping checkpoint push")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(
            folder_path=local_dir,
            path_in_repo=f"checkpoints/{subfolder}",
            repo_id=repo,
            repo_type="model",
        )
        logger.info("Pushed checkpoint → %s/checkpoints/%s", repo, subfolder)
    except Exception as e:
        logger.error("Hub checkpoint push failed: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_environment()
    cfg = build_config()
    stage = cfg["stage"]

    logger.info("=" * 60)
    logger.info("PS2 — RL-Enhanced SLM Reasoning")
    logger.info("Stage: %s", stage)
    logger.info("Model: %s", cfg["model"]["base_id"])
    logger.info("=" * 60)

    if stage == "baseline":
        scores = run_baseline(cfg)
        logger.info("Done. GSM8K baseline: %.2f%%", scores["gsm8k_zero_shot_pct"])

    elif stage == "sft":
        ckpt = run_sft(cfg)
        logger.info("Done. SFT checkpoint: %s", ckpt)

    elif stage == "grpo":
        ckpt = run_grpo(cfg, use_curriculum=False)
        logger.info("Done. GRPO checkpoint: %s", ckpt)

    elif stage == "curriculum":
        ckpt = run_grpo(cfg, use_curriculum=True)
        logger.info("Done. Curriculum GRPO checkpoint: %s", ckpt)

    else:
        logger.error("Unknown TRAIN_STAGE='%s'. Must be one of: baseline, sft, grpo, curriculum", stage)
        sys.exit(1)


if __name__ == "__main__":
    main()