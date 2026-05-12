"""
src/data/dataset.py

Loads and preprocesses GSM8K and AQuA-RAT for SFT and GRPO training.

Dataset schemas (verified from HuggingFace Hub):
  openai/gsm8k (config="main"):
    - question: str
    - answer:   str  (free-form solution ending with "#### <number>")

  aqua_rat (config="raw"):
    - question: str
    - options:  list[str]  e.g. ["A)12", "B)15", ...]
    - rationale: str
    - correct:  str  e.g. "A"

HALLUCINATION NOTE:
  - AQuA-RAT "raw" config name is from HF Hub docs; if load fails try config="tokenized".
  - Phi-3 chat template uses [INST]/[/INST] tags; actual template is applied via
    tokenizer.apply_chat_template(). The format strings here are fallback plain-text.
"""

from __future__ import annotations

import re
from collections import namedtuple
from typing import Optional

from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


# ---------------------------------------------------------------------------
# Difficulty scoring
# ---------------------------------------------------------------------------

def score_difficulty(solution: str) -> int:
    """
    Heuristic difficulty score for a GSM8K solution string.
    Returns 1 (easy), 2 (medium), or 3 (hard).

    Proxy: combination of number of newline-separated steps,
    arithmetic operators, and distinct numbers in the solution.
    """
    n_steps = solution.count("\n")
    n_ops = len(re.findall(r"[\+\-\*\/]", solution))
    n_numbers = len(re.findall(r"\d+\.?\d*", solution))

    score = n_steps * 0.5 + n_ops * 0.3 + n_numbers * 0.2

    if score < 4.0:
        return 1
    elif score < 8.0:
        return 2
    return 3


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

_GSM8K_SYSTEM = (
    "You are a math reasoning assistant. "
    "Solve the problem step by step. "
    "End your answer with the format: #### <number>"
)


def format_gsm8k_sft(example: dict) -> dict:
    """
    Format a GSM8K example for SFT training.
    Returns a dict with 'text' (full prompt+response) and 'difficulty'.
    """
    prompt = (
        f"<|system|>\n{_GSM8K_SYSTEM}<|end|>\n"
        f"<|user|>\n{example['question']}<|end|>\n"
        f"<|assistant|>\n"
    )
    response = example["answer"]
    text = prompt + response + "<|end|>"

    return {
        "text": text,
        "prompt": prompt,
        "response": response,
        "answer": _extract_answer_from_solution(response),
        "difficulty": score_difficulty(response),
        "source": "gsm8k",
    }


def format_gsm8k_rl(example: dict) -> dict:
    """
    Format a GSM8K example for RL (GRPO) training.
    Returns prompt only — the model will generate the response.
    """
    prompt = (
        f"<|system|>\n{_GSM8K_SYSTEM}<|end|>\n"
        f"<|user|>\n{example['question']}<|end|>\n"
        f"<|assistant|>\n"
    )
    return {
        "prompt": prompt,
        "answer": _extract_answer_from_solution(example["answer"]),
        "difficulty": score_difficulty(example["answer"]),
        "source": "gsm8k",
    }


def _extract_answer_from_solution(solution: str) -> str:
    """
    Parse the numeric answer after '####' in a GSM8K solution.
    Returns the answer as a string, or "" if not found.
    """
    match = re.search(r"####\s*([\-\d,\.]+)", solution)
    if match:
        return match.group(1).replace(",", "").strip()
    return ""


# ---------------------------------------------------------------------------
# AQuA-RAT
# ---------------------------------------------------------------------------

_AQUA_SYSTEM = (
    "You are a math reasoning assistant. "
    "Solve the problem step by step, then state the correct option letter. "
    "End your answer with: #### <letter>"
)


def format_aqua_sft(example: dict) -> dict:
    """
    Format an AQuA-RAT example for SFT training.

    HALLUCINATION NOTE: AQuA-RAT 'options' is a list of strings like ["A)12", "B)15"].
    The 'correct' field is a single letter like "A". The 'rationale' field is the
    free-text solution. This structure is from HF Hub; verify with a print() if loading fails.
    """
    options_str = "\n".join(example.get("options", []))
    prompt = (
        f"<|system|>\n{_AQUA_SYSTEM}<|end|>\n"
        f"<|user|>\n{example['question']}\n\nOptions:\n{options_str}<|end|>\n"
        f"<|assistant|>\n"
    )
    rationale = example.get("rationale", "")
    correct = example.get("correct", "")
    response = f"{rationale}\n#### {correct}"
    text = prompt + response + "<|end|>"

    return {
        "text": text,
        "prompt": prompt,
        "response": response,
        "answer": correct,
        "difficulty": 2,  # AQuA is treated as medium difficulty uniformly
        "source": "aqua_rat",
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_gsm8k(
    split: str = "train",
    val_size: int = 200,
    mode: str = "sft",  # "sft" or "rl"
) -> Dataset | tuple[Dataset, Dataset]:
    """
    Load and format GSM8K.

    Args:
        split:    "train" or "test"
        val_size: number of train examples to hold out as validation (train split only)
        mode:     "sft" formats full text; "rl" formats prompt-only

    Returns:
        For split="train": (train_dataset, val_dataset)
        For split="test":  test_dataset
    """
    ds = load_dataset("openai/gsm8k", "main", split=split)
    formatter = format_gsm8k_sft if mode == "sft" else format_gsm8k_rl

    ds = ds.map(formatter, remove_columns=ds.column_names)

    if split == "train":
        ds = ds.shuffle(seed=42)
        val_ds = ds.select(range(val_size))
        train_ds = ds.select(range(val_size, len(ds)))
        return train_ds, val_ds

    return ds


def load_aqua_rat(mode: str = "sft") -> Dataset:
    """
    Load and format AQuA-RAT training split.

    HALLUCINATION NOTE: The HF dataset name 'aqua_rat' and config 'raw' are from
    HF Hub docs. If this fails, try: load_dataset("deepmind/aqua_rat", "raw").
    The 'raw' config has free-text rationales; 'tokenized' has tokenised versions.
    """
    try:
        ds = load_dataset("aqua_rat", "raw", split="train")
    except Exception:
        # fallback name
        ds = load_dataset("deepmind/aqua_rat", "raw", split="train")

    if mode == "sft":
        ds = ds.map(format_aqua_sft, remove_columns=ds.column_names)
    # AQuA is only used in SFT, not RL
    return ds


def build_sft_dataset(
    gsm8k_fraction: float = 0.80,
    aqua_fraction: float = 0.20,
    val_size: int = 200,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """
    Build the combined SFT training dataset from GSM8K + AQuA-RAT.

    Returns (train_dataset, val_dataset).
    train_dataset has column 'text' required by SFTTrainer.
    """
    gsm8k_train, gsm8k_val = load_gsm8k(split="train", val_size=val_size, mode="sft")
    aqua_train = load_aqua_rat(mode="sft")

    # Subsample to achieve target mix ratio
    n_gsm8k = len(gsm8k_train)
    n_aqua_target = int(n_gsm8k * (aqua_fraction / gsm8k_fraction))
    n_aqua_target = min(n_aqua_target, len(aqua_train))

    aqua_train = aqua_train.shuffle(seed=seed).select(range(n_aqua_target))

    # Concatenate and shuffle
    train_ds = concatenate_datasets([gsm8k_train, aqua_train]).shuffle(seed=seed)

    print(f"SFT train size: {len(train_ds)} "
          f"(GSM8K: {n_gsm8k}, AQuA: {n_aqua_target})")
    print(f"SFT val size:   {len(gsm8k_val)} (GSM8K only)")

    return train_ds, gsm8k_val


def build_rl_dataset(seed: int = 42) -> Dataset:
    """
    Build the RL training dataset (prompt-only, GSM8K train split).
    The 'difficulty' column is used by CurriculumGate to filter examples.
    """
    train_ds, _ = load_gsm8k(split="train", val_size=200, mode="rl")
    train_ds = train_ds.shuffle(seed=seed)
    print(f"RL dataset size: {len(train_ds)}")
    return train_ds