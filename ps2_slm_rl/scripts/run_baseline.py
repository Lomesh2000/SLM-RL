"""
scripts/run_baseline.py

Week 1 task: Record baseline Phi-3-mini performance on GSM8K before any training.
This is the "Baseline" column in the ablation table.

Run:
    python scripts/run_baseline.py --limit 500

Saves baseline scores to results/baseline.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="microsoft/Phi-3-mini-4k-instruct")
    parser.add_argument("--limit", type=int, default=500,
                        help="Number of GSM8K examples to evaluate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    from lm_eval import simple_evaluate

    logger.info("Running baseline eval on %s, limit=%d", args.model_id, args.limit)

    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model_id},trust_remote_code=True",
        tasks=["gsm8k"],
        num_fewshot=0,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
        device=args.device,
    )

    gsm8k_results = results["results"]["gsm8k"]
    logger.info("GSM8K results: %s", gsm8k_results)

    # Try both possible metric keys
    score = (
        gsm8k_results.get("exact_match,flexible-extract")
        or gsm8k_results.get("exact_match,strict-match")
        or gsm8k_results.get("acc,none")
    )

    print(f"\nBaseline GSM8K score: {score*100:.2f}%")
    print("Full result keys:", list(gsm8k_results.keys()))

    os.makedirs("results", exist_ok=True)
    with open("results/baseline.json", "w") as f:
        json.dump({"baseline_gsm8k": score, "raw": gsm8k_results}, f, indent=2)
    logger.info("Saved to results/baseline.json")


if __name__ == "__main__":
    main()