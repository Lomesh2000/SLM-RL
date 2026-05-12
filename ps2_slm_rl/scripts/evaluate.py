"""
scripts/evaluate.py

Evaluation script using lm-evaluation-harness (lm_eval==0.4.3).

Run:
    # Evaluate a merged model (no LoRA adapter):
    python scripts/evaluate.py --model-path ./final_model --tasks gsm8k mmlu

    # Evaluate a LoRA checkpoint on top of base model:
    python scripts/evaluate.py \
        --model-path microsoft/Phi-3-mini-4k-instruct \
        --peft-path ./checkpoints/grpo \
        --tasks gsm8k mmlu strategy_qa

    # Run full ablation across all checkpoints:
    python scripts/evaluate.py --ablation

HALLUCINATION NOTES:
  - lm_eval.simple_evaluate() signature verified for lm_eval==0.4.3.
    Key params: model, model_args, tasks, num_fewshot, batch_size, limit, log_samples.
  - Task names in lm_eval 0.4.3:
      GSM8K:      "gsm8k"
      MMLU:       "mmlu" (evaluates all MMLU subsets, averages)
      StrategyQA: "strategy_qa"
    These names were verified from lm_eval 0.4.3 task registry. If a task name
    fails, run: lm_eval --tasks list | grep -i <name>
  - MMLU num_fewshot=5 is standard practice from the original MMLU paper.
  - model_args string format for PEFT: "pretrained=<base>,peft=<adapter_path>"
    This is the documented lm_eval HF model_args format.
  - 'limit' parameter accepts int (number of examples) or float (fraction).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


TASK_FEWSHOT = {
    "gsm8k": 0,
    "mmlu": 5,
    "strategy_qa": 0,
}

# Metric keys to extract from lm_eval results
# HALLUCINATION NOTE: these metric key names are from lm_eval 0.4.3 output format.
# GSM8K uses 'exact_match,strict-match' or 'exact_match,flexible-extract'.
# MMLU uses 'acc,none'. StrategyQA uses 'acc,none'.
# If keys differ, print(results['results']) to inspect.
TASK_METRIC_KEY = {
    "gsm8k": "exact_match,flexible-extract",
    "mmlu": "acc,none",
    "strategy_qa": "acc,none",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SLM checkpoints")
    parser.add_argument("--model-path", default="microsoft/Phi-3-mini-4k-instruct",
                        help="Path to merged model or HF model ID")
    parser.add_argument("--peft-path", default=None,
                        help="Path to LoRA adapter checkpoint (optional)")
    parser.add_argument("--tasks", nargs="+",
                        default=["gsm8k", "mmlu", "strategy_qa"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples per task (for quick testing)")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--run-name", default="eval",
                        help="Name for the output JSON file")
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation across all standard checkpoints")
    parser.add_argument("--device", default="cuda",
                        help="Device: 'cuda', 'cuda:0', 'cpu'")
    return parser.parse_args()


def build_model_args(model_path: str, peft_path: str | None = None) -> str:
    """Build the model_args string for lm_eval simple_evaluate."""
    args = f"pretrained={model_path}"
    if peft_path:
        args += f",peft={peft_path}"
    args += ",trust_remote_code=True"
    return args


def run_evaluation(
    model_path: str,
    peft_path: str | None,
    tasks: list[str],
    batch_size: int = 4,
    limit: int | None = None,
    device: str = "cuda",
) -> dict:
    """
    Run lm_eval evaluation and return results dict.

    Returns a flat dict: {task_name: score, ...}
    """
    from lm_eval import simple_evaluate

    model_args = build_model_args(model_path, peft_path)
    logger.info("Evaluating: %s", model_args)
    logger.info("Tasks: %s", tasks)

    # num_fewshot: pass per-task as a list aligned with tasks
    num_fewshot = [TASK_FEWSHOT.get(t, 0) for t in tasks]

    raw_results = simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
        device=device,
    )

    # Extract primary metric per task
    scores = {}
    for task in tasks:
        if task not in raw_results["results"]:
            logger.warning("Task '%s' not found in results.", task)
            continue
        task_results = raw_results["results"][task]
        metric_key = TASK_METRIC_KEY.get(task)
        if metric_key and metric_key in task_results:
            scores[task] = task_results[metric_key]
        else:
            # Fallback: print all available keys so user can find the right one
            logger.warning(
                "Metric key '%s' not found for task '%s'. Available: %s",
                metric_key, task, list(task_results.keys()),
            )
            # Try common fallbacks
            for fallback in ["acc,none", "exact_match,flexible-extract",
                             "exact_match,strict-match"]:
                if fallback in task_results:
                    scores[task] = task_results[fallback]
                    logger.info("Used fallback metric '%s' for task '%s'", fallback, task)
                    break

    return scores, raw_results


def save_results(
    scores: dict,
    raw_results: dict,
    output_dir: str,
    run_name: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{run_name}.json")
    with open(out_path, "w") as f:
        json.dump({"scores": scores, "raw": raw_results["results"]}, f, indent=2)
    logger.info("Results saved to %s", out_path)


def run_ablation(tasks: list[str], batch_size: int, limit: int | None, device: str):
    """
    Run evaluation across all standard checkpoints and print the ablation table.

    Checkpoints expected (adjust paths to match your training output):
      - Baseline:          microsoft/Phi-3-mini-4k-instruct (no peft)
      - SFT:               checkpoints/sft (LoRA on base)
      - GRPO:              checkpoints/grpo (LoRA on base)
      - Curriculum GRPO:   checkpoints/curriculum_grpo (LoRA on base)
    """
    base_model = "microsoft/Phi-3-mini-4k-instruct"
    runs = [
        ("Baseline",          base_model,  None),
        ("SFT",               base_model,  "checkpoints/sft"),
        ("SFT + GRPO",        base_model,  "checkpoints/grpo"),
        ("SFT + Curriculum",  base_model,  "checkpoints/curriculum_grpo"),
    ]

    all_scores = []
    for name, model_path, peft_path in runs:
        if peft_path and not os.path.exists(peft_path):
            logger.warning("Checkpoint not found: %s — skipping %s", peft_path, name)
            continue
        logger.info("Evaluating: %s", name)
        scores, raw = run_evaluation(model_path, peft_path, tasks, batch_size, limit, device)
        scores["run"] = name
        all_scores.append(scores)
        save_results(scores, raw, "results", name.replace(" ", "_").lower())

    if all_scores:
        df = pd.DataFrame(all_scores).set_index("run")
        # Format as percentage
        for col in df.columns:
            df[col] = (df[col] * 100).round(2).astype(str) + "%"
        print("\n" + "="*60)
        print("ABLATION TABLE")
        print("="*60)
        print(df.to_string())
        print("="*60 + "\n")
        df.to_csv("results/ablation_table.csv")
        logger.info("Ablation table saved to results/ablation_table.csv")


def main():
    args = parse_args()

    if args.ablation:
        run_ablation(
            tasks=args.tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            device=args.device,
        )
        return

    scores, raw_results = run_evaluation(
        model_path=args.model_path,
        peft_path=args.peft_path,
        tasks=args.tasks,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
    )

    print("\n=== SCORES ===")
    for task, score in scores.items():
        print(f"  {task:20s}: {score*100:.2f}%")

    save_results(scores, raw_results, args.output_dir, args.run_name)


if __name__ == "__main__":
    main()