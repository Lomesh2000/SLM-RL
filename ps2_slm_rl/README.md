# PS2 — RL-Enhanced SLM Reasoning

Fine-tunes **Phi-3-Mini (3.8B)** with a two-stage pipeline:
1. **SFT warm-up** on GSM8K + AQuA-RAT with chain-of-thought formatting
2. **GRPO** (Group Relative Policy Optimisation) with a curriculum-gated
   difficulty progression and a composite reward (outcome + CoT process + format)

Target: improve GSM8K from ~45% → ≥50% (+5pp) and StrategyQA ≥65%.

---

## Project structure

```
ps2_slm_rl/
├── configs/
│   ├── sft.yaml              # SFT hyperparameters
│   ├── grpo.yaml             # GRPO + reward settings
│   └── curriculum.yaml       # Curriculum gate thresholds
├── src/
│   ├── data/dataset.py       # GSM8K + AQuA-RAT loaders + difficulty scoring
│   ├── model/loader.py       # Model loading, LoRA wrapping, merge
│   ├── rewards/combined.py   # Composite reward function (trl 1.4.0 compatible)
│   └── training/curriculum.py # CurriculumGate class
├── scripts/
│   ├── run_baseline.py       # Week 1: record baseline before training
│   ├── train_sft.py          # Week 2: SFT warm-up
│   ├── train_grpo.py         # Week 3-4: GRPO with optional curriculum
│   ├── evaluate.py           # Week 5: full eval + ablation table
│   ├── benchmark_latency.py  # Latency measurement
│   └── merge_lora.py         # Merge LoRA into base for inference
└── notebooks/
    └── kaggle_pipeline.ipynb # Kaggle entry point (P100-aware)
```

---

## Quickstart

```bash
pip install -r requirements.txt
```

### Week 1 — Baseline
```bash
python scripts/run_baseline.py --limit 500
```

### Week 2 — SFT

Verify the correct LoRA `target_modules` for your model first:
```bash
python src/model/loader.py --print-modules
```

Then train:
```bash
python scripts/train_sft.py --config configs/sft.yaml
```

### Week 3-4 — GRPO with curriculum
```bash
python scripts/train_grpo.py \
    --config configs/grpo.yaml \
    --curriculum-config configs/curriculum.yaml
```

Without curriculum (baseline GRPO):
```bash
python scripts/train_grpo.py --config configs/grpo.yaml --curriculum-config none
```

### Week 5 — Full evaluation + ablation table
```bash
python scripts/evaluate.py --ablation --tasks gsm8k mmlu strategy_qa --batch-size 4
```

### Merge LoRA for inference
```bash
python scripts/merge_lora.py \
    --base-model microsoft/Phi-3-mini-4k-instruct \
    --lora-path checkpoints/curriculum_grpo \
    --output-path final_model
```

### Latency benchmark
```bash
python scripts/benchmark_latency.py --model-path ./final_model
```

---

## Kaggle P100 settings

Edit `configs/sft.yaml` and `configs/grpo.yaml`:
```yaml
model:
  torch_dtype: float16        # NOT bfloat16 (P100 doesn't support it)
  attn_implementation: eager  # NOT flash_attention_2 (not supported on P100)

training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

---

## Verified library versions

| Package | Version | Notes |
|---|---|---|
| trl | 1.4.0 | GRPOTrainer, GRPOConfig, SFTTrainer, SFTConfig |
| peft | 0.19.1 | LoraConfig, PeftModel |
| transformers | 4.57.6 | AutoModelForCausalLM |
| lm_eval | 0.4.3 | simple_evaluate |

All API calls in this codebase use field names verified against these versions.

---

## Hallucination disclosures

These items are **not verified** and should be double-checked before running:

1. **Phi-3-mini `target_modules`**: `gate_up_proj` is listed as the fused
   projection based on architecture docs. Run `--print-modules` to confirm.
   If training fails with "target module not found", adjust this list.

2. **AQuA-RAT dataset config**: `load_dataset("aqua_rat", "raw")` is from
   HF Hub documentation. If it fails, try `"deepmind/aqua_rat"` as the
   dataset name. The code includes a fallback.

3. **lm_eval task metric keys**: GSM8K uses `exact_match,flexible-extract`.
   If this key is missing from results, the code logs available keys so you
   can update `TASK_METRIC_KEY` in `scripts/evaluate.py`.

4. **Step-level process reward**: `count_correct_steps()` is a heuristic
   proxy. A real process reward model (PRM) trained on annotated data would
   be more accurate but requires separate training data (e.g. PRM800K).

5. **Curriculum stage stepping**: The stage-by-stage training approach uses
   `max_steps` override per stage. This is standard HuggingFace behaviour
   but means steps are estimated, not exact.

6. **Expected accuracy numbers** in the ablation table template in the
   interactive plan are illustrative targets, not guaranteed results.
   Actual numbers depend on hardware, random seed, and exact dataset splits.

---

## Ablation table (fill in after Week 5)

| Model | GSM8K | MMLU | StrategyQA | Latency (ms/q) |
|---|---|---|---|---|
| Phi-3-Mini base | — | — | — | — |
| + SFT | — | — | — | — |
| + SFT + GRPO | — | — | — | — |
| + SFT + Curriculum GRPO | — | — | — | — |

---

## W&B metrics to monitor

| Metric | Expected behaviour |
|---|---|
| `reward/mean` | Trending upward through training |
| `kl_div` | Stays between 0.5–3.0; spikes → raise `beta` |
| `entropy` | Should not collapse to 0 |
| `curriculum/stage` | Steps from 0 → 1 → 2 across training |
| `curriculum/rolling_acc` | Rises within each stage |