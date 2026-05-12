"""
scripts/benchmark_latency.py

Measures inference latency for a given model checkpoint.
Reports median, P95, and tokens/sec.

Run:
    python scripts/benchmark_latency.py --model-path ./final_model
    python scripts/benchmark_latency.py \
        --model-path microsoft/Phi-3-mini-4k-instruct \
        --peft-path checkpoints/curriculum_grpo
"""

from __future__ import annotations

import argparse
import logging
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 20 representative GSM8K-style problems for benchmarking
BENCHMARK_QUESTIONS = [
    "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the remainder at $2 each. How much does she make every day?",
    "A store had 150 apples. They sold 40% on Monday and 30% of the remaining on Tuesday. How many apples are left?",
    "If a train travels 60 mph for 2.5 hours, how many miles does it travel?",
    "There are 30 students. 40% are boys. How many girls are there?",
    "A rectangle has length 12 and width 8. What is the perimeter?",
    "John has $200. He spends $45 on books and $30 on food. How much is left?",
    "A class scored 85, 90, 78, 92, and 88 on tests. What is the average?",
    "If 6 workers can complete a job in 10 days, how long would 4 workers take?",
    "A tank fills at 15 liters/min and leaks at 5 liters/min. How long to fill 200 liters?",
    "Maria earns $15/hr and works 8 hours Monday-Friday. What are her weekly earnings?",
    "A shirt costs $40 with 25% discount. What is the final price?",
    "Two pipes fill a tank: pipe A in 4 hours, pipe B in 6 hours. Together, how long?",
    "A car depreciates 15% per year from $20000. What is its value after 2 years?",
    "If 3x + 7 = 22, what is x?",
    "A pizza has 8 slices. 3 friends eat 2 slices each. What fraction remains?",
    "Sam runs 5 km in 25 mins. At the same pace, how long to run 8 km?",
    "A box has 24 chocolates. 1/3 are dark, 1/4 are white, rest are milk. How many milk?",
    "An item costs $80. With 10% tax, what is the total?",
    "A rectangle's area is 96. Length is 12. What is the width?",
    "If 5 items cost $37.50, what does 1 item cost?",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark inference latency")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--peft-path", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--n-warmup", type=int, default=3,
                        help="Warmup runs (not counted in stats)")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model_and_tok(model_path: str, peft_path: str | None, device: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device
    )

    if peft_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, peft_path)
        logger.info("Loaded LoRA adapter from %s", peft_path)

    model.eval()
    return model, tok


def run_benchmark(
    model,
    tok,
    questions: list[str],
    max_new_tokens: int,
    n_warmup: int,
    device: str,
) -> dict:
    times_ms = []
    tokens_generated = []

    all_questions = questions[:n_warmup] + questions  # warmup prepended
    is_warmup = [True] * n_warmup + [False] * len(questions)

    with torch.inference_mode():
        for question, warmup in zip(all_questions, is_warmup):
            prompt = (
                f"<|system|>\nSolve step by step. End with: #### <number><|end|>\n"
                f"<|user|>\n{question}<|end|>\n"
                f"<|assistant|>\n"
            )
            inputs = tok(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            if warmup:
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False, pad_token_id=tok.eos_token_id)
                continue

            t0 = time.perf_counter()
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000

            n_new_tokens = output.shape[1] - input_len
            times_ms.append(elapsed_ms)
            tokens_generated.append(n_new_tokens)

    median_ms = statistics.median(times_ms)
    p95_ms = sorted(times_ms)[int(0.95 * len(times_ms))]
    mean_tokens = statistics.mean(tokens_generated)
    tokens_per_sec = mean_tokens / (median_ms / 1000)

    return {
        "n_samples": len(times_ms),
        "median_latency_ms": round(median_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "mean_new_tokens": round(mean_tokens, 1),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "all_times_ms": [round(t, 2) for t in times_ms],
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        logger.warning("CUDA not available, switching to CPU (will be slow)")
        args.device = "cpu"

    logger.info("Loading model from %s", args.model_path)
    model, tok = load_model_and_tok(args.model_path, args.peft_path, args.device)

    logger.info(
        "Running benchmark: %d questions, %d warmup, max_new_tokens=%d",
        len(BENCHMARK_QUESTIONS), args.n_warmup, args.max_new_tokens,
    )

    results = run_benchmark(
        model, tok, BENCHMARK_QUESTIONS,
        args.max_new_tokens, args.n_warmup, args.device,
    )

    print("\n=== LATENCY BENCHMARK ===")
    print(f"  Samples:          {results['n_samples']}")
    print(f"  Median latency:   {results['median_latency_ms']} ms")
    print(f"  P95 latency:      {results['p95_latency_ms']} ms")
    print(f"  Mean new tokens:  {results['mean_new_tokens']}")
    print(f"  Tokens/sec:       {results['tokens_per_sec']}")
    print("=" * 26 + "\n")

    import json
    out_path = "results/latency_benchmark.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()