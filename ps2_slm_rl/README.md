---
license: mit
tags:
  - reinforcement-learning
  - small-language-model
  - grpo
  - curriculum-learning
  - math-reasoning
  - phi-3
language:
  - en
datasets:
  - openai/gsm8k
  - aqua_rat
base_model:
  - microsoft/Phi-3-mini-4k-instruct
---

# PS2 — RL-Enhanced SLM Reasoning (Phi-3-Mini + GRPO + Curriculum)

Fine-tunes **Phi-3-Mini (3.8B)** with a two-stage pipeline:
1. **SFT warm-up** on GSM8K + AQuA-RAT with chain-of-thought formatting
2. **GRPO** with curriculum-gated difficulty progression and composite reward

---

## Training stages

Run each as a separate HF Training Job by setting the `TRAIN_STAGE` env var:

| `TRAIN_STAGE` | What it does | Run first? |
|---|---|---|
| `baseline` | Records Phi-3-Mini zero-shot GSM8K score | ✅ Yes |
| `sft` | SFT warm-up with LoRA on GSM8K + AQuA-RAT | After baseline |
| `grpo` | GRPO training (no curriculum) | After sft |
| `curriculum` | GRPO with curriculum gating | After sft |

---

## Environment variables

### Required
| Variable | Description |
|---|---|
| `HF_TOKEN` | Your HuggingFace token (model download + checkpoint push) |
| `TRAIN_STAGE` | One of: `baseline`, `sft`, `grpo`, `curriculum` |

### Optional
| Variable | Default | Description |
|---|---|---|
| `OUTPUT_REPO` | _(empty)_ | `username/repo-name` to push checkpoints/results |
| `WANDB_API_KEY` | _(empty)_ | W&B key; training runs without it (logging disabled) |
| `BASE_MODEL_ID` | `microsoft/Phi-3-mini-4k-instruct` | Model to fine-tune |
| `EVAL_LIMIT` | `500` | GSM8K examples for baseline eval |
| `BATCH_SIZE` | `1` | Per-device train batch size |
| `GRAD_ACCUM` | `16` | Gradient accumulation steps |
| `LR` | `2e-4` (SFT) / `1e-5` (GRPO) | Learning rate |
| `NUM_EPOCHS` | `2` (SFT) / `1` (GRPO) | Training epochs |
| `LORA_R` | `16` | LoRA rank |
| `GRPO_BETA` | `0.05` | KL penalty coefficient |
| `GRPO_NUM_GENERATIONS` | `8` | Completions sampled per prompt |
| `HF_HUB_ENABLE_HF_TRANSFER` | `1` | Fast model download (keep 1) |

---

## Recommended hardware

| Stage | Minimum | Recommended |
|---|---|---|
| baseline | A10G (24GB) | A100 40GB |
| sft | A100 40GB | A100 80GB |
| grpo / curriculum | A100 40GB | A100 80GB or 2×A100 |

---

## File structure

```
train.py               ← HF Training Job entrypoint (this is what HF runs)
requirements.txt       ← pinned dependencies
src/
  data/dataset.py      ← GSM8K + AQuA-RAT loaders
  model/loader.py      ← model loading, LoRA, merge
  rewards/combined.py  ← GRPO reward functions
  training/curriculum.py ← CurriculumGate
scripts/
  run_baseline.py      ← standalone baseline script
  train_sft.py         ← standalone SFT script
  train_grpo.py        ← standalone GRPO script
  evaluate.py          ← full eval + ablation table
  benchmark_latency.py ← latency measurement
  merge_lora.py        ← merge LoRA for inference
configs/
  sft.yaml             ← SFT hyperparams
  grpo.yaml            ← GRPO hyperparams
  curriculum.yaml      ← curriculum gate config
```

---

## Verified dependency versions

| Package | Pinned version | Notes |
|---|---|---|
| trl | 1.4.0 | GRPOConfig fields: `beta`, `max_completion_length`, `epsilon` |
| peft | 0.19.1 | PeftModel.from_pretrained signature |
| transformers | ≥4.56.2 | trl 1.4.0 minimum; tested on 5.8.0 |
| datasets | ≥4.7.0 | trl 1.4.0 minimum; tested on 4.8.5 |
| lm_eval | 0.4.3 | task names and metric keys verified |

---

## Hallucination disclosures

Items that are **not runtime-verified** and should be checked on first run:

1. **Phi-3-mini `target_modules`** — `gate_up_proj` from architecture docs.
   Run `python src/model/loader.py --print-modules` to verify.
2. **AQuA-RAT dataset name** — fallback from `aqua_rat` → `deepmind/aqua_rat` included.
3. **lm_eval GSM8K metric key** — code tries all known variants and logs the actual key found.
4. **Process reward** — rule-based CoT heuristic, not a trained PRM.