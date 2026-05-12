"""
src/rewards/combined.py

Reward functions for GRPO training.

CRITICAL: The GRPOTrainer in trl==1.4.0 calls reward functions with this signature:
    reward_func(prompts, completions, completion_ids, **reward_kwargs)

Where reward_kwargs contains all dataset columns EXCEPT 'prompt', 'completion',
and 'completion_ids'. So if your dataset has an 'answer' column, it arrives
as reward_kwargs['answer'].

This was verified by reading GRPOTrainer._calculate_rewards source in trl 1.4.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_answer(text: str) -> str | None:
    """
    Extract the numeric answer after the '####' marker.
    Returns None if no valid answer found.

    GSM8K format: "... step N. #### 42"
    """
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        # Normalise: remove commas, strip whitespace
        raw = match.group(1).replace(",", "").strip()
        # Remove trailing punctuation
        raw = raw.rstrip(".")
        return raw
    return None


def normalise_number(s: str) -> str | None:
    """Normalise a numeric string for comparison (handle floats, negatives)."""
    try:
        val = float(s.replace(",", ""))
        # Compare as int if it's a whole number
        if val == int(val):
            return str(int(val))
        return f"{val:.4f}"
    except (ValueError, TypeError):
        return None


def answers_match(pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    # Exact string match first
    if pred.strip() == gold.strip():
        return True
    # Numeric match
    norm_pred = normalise_number(pred)
    norm_gold = normalise_number(gold)
    if norm_pred is not None and norm_gold is not None:
        return norm_pred == norm_gold
    return False


# ---------------------------------------------------------------------------
# Process reward helpers
# ---------------------------------------------------------------------------

def has_valid_cot(text: str, min_steps: int = 2) -> bool:
    """
    Check whether the completion has a plausible chain-of-thought before '####'.

    Heuristic: at least `min_steps` non-trivial sentences (>10 chars) in the
    reasoning portion before the final answer marker.

    HALLUCINATION NOTE: A "learned" process reward model (PRM) would be more
    accurate but requires a separately trained verifier. This rule-based version
    is a proxy that works well enough for initial RL training.
    """
    parts = text.split("####")
    if len(parts) < 2:
        return False
    reasoning = parts[0]
    sentences = [
        s.strip()
        for s in re.split(r"[.\n]", reasoning)
        if len(s.strip()) > 10
    ]
    return len(sentences) >= min_steps


def count_correct_steps(completion: str, gold_answer: str) -> int:
    """
    Count intermediate numeric values in the completion that appear in the
    gold solution. This is a very rough proxy for step-level correctness.

    HALLUCINATION NOTE: True step-level reward requires a PRM trained on
    process-annotated data (e.g. PRM800K). This is a heuristic approximation.
    Returns 0 or 1 (not a real step count).
    """
    # Extract all numbers from the completion
    completion_numbers = set(
        re.findall(r"\b\d+\.?\d*\b", completion.split("####")[0])
    )
    # If the gold answer appears as an intermediate calculation, give partial credit
    gold_norm = normalise_number(gold_answer)
    if gold_norm and gold_norm in completion_numbers:
        return 1
    return 0


def is_valid_format(text: str) -> bool:
    """Check the completion ends with a #### marker."""
    return bool(re.search(r"####\s*[\-\d]", text))


# ---------------------------------------------------------------------------
# Reward config
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    outcome_weight: float = 1.0
    process_weight: float = 0.3
    step_weight: float = 0.1
    format_penalty: float = -0.1
    reward_cap: float = 1.5


# ---------------------------------------------------------------------------
# Reward functions — compatible with GRPOTrainer._calculate_rewards
# ---------------------------------------------------------------------------

def make_reward_fn(cfg: RewardConfig | None = None):
    """
    Factory that returns a reward function with the correct trl 1.4.0 signature.

    The returned function signature is:
        fn(prompts, completions, completion_ids, **reward_kwargs) -> list[float]

    reward_kwargs will contain the dataset columns (e.g. 'answer', 'difficulty').
    """
    if cfg is None:
        cfg = RewardConfig()

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        **reward_kwargs: Any,
    ) -> list[float]:
        """
        Composite reward: outcome + process + step + format.

        reward_kwargs['answer'] is populated automatically by GRPOTrainer
        from the dataset's 'answer' column.
        """
        gold_answers: list[str] = reward_kwargs.get("answer", [""] * len(completions))

        rewards = []
        for completion, gold in zip(completions, gold_answers):
            r = 0.0

            # 1. Outcome reward: exact answer match
            pred = extract_answer(completion)
            if answers_match(pred, gold):
                r += cfg.outcome_weight

            # 2. Process reward: valid CoT structure present
            if has_valid_cot(completion):
                r += cfg.process_weight

            # 3. Step-level partial credit (heuristic)
            r += cfg.step_weight * count_correct_steps(completion, gold)

            # 4. Format penalty: no #### marker
            if not is_valid_format(completion):
                r += cfg.format_penalty  # negative value

            # Cap to prevent outliers from dominating
            rewards.append(min(max(r, cfg.format_penalty), cfg.reward_cap))

        return rewards

    return reward_fn


# ---------------------------------------------------------------------------
# Standalone outcome-only reward (for ablation)
# ---------------------------------------------------------------------------

def make_outcome_only_reward_fn():
    """Reward function using only exact-match outcome. Used in ablation studies."""

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        completion_ids: list[list[int]],
        **reward_kwargs: Any,
    ) -> list[float]:
        gold_answers: list[str] = reward_kwargs.get("answer", [""] * len(completions))
        return [
            1.0 if answers_match(extract_answer(c), g) else 0.0
            for c, g in zip(completions, gold_answers)
        ]

    return reward_fn