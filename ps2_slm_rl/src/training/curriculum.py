"""
src/training/curriculum.py

CurriculumGate: tracks rolling accuracy and gates training to progressively
harder examples as the model improves.

This implements difficulty-gated curriculum inside the RL training loop,
as opposed to the more common approach of pre-sorting data before training.
The gate fires when rolling accuracy over a window exceeds a threshold,
then opens the next difficulty tier.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import wandb

logger = logging.getLogger(__name__)


@dataclass
class CurriculumConfig:
    window_size: int = 200
    min_window_before_advance: int = 200
    easy_to_medium_threshold: float = 0.65
    medium_to_hard_threshold: float = 0.60
    # Stage definitions: which difficulty ints to include at each stage
    stage_difficulties: list[list[int]] = field(
        default_factory=lambda: [[1], [1, 2], [1, 2, 3]]
    )
    stage_names: list[str] = field(
        default_factory=lambda: ["easy", "medium", "hard"]
    )


class CurriculumGate:
    """
    Tracks rolling accuracy and advances curriculum stage when ready.

    Usage in training loop:
        gate = CurriculumGate(cfg)
        for batch in dataloader:
            if batch['difficulty'] not in gate.allowed_difficulties:
                continue
            # ... GRPO step ...
            correct = check_answer(pred, gold)
            gate.update(correct)
            gate.maybe_advance(global_step)
    """

    def __init__(self, cfg: CurriculumConfig | None = None):
        self.cfg = cfg or CurriculumConfig()
        self.stage: int = 0
        self.history: deque[float] = deque(maxlen=self.cfg.window_size)
        self._total_updates: int = 0
        self._stage_transitions: list[dict] = []  # log for reporting

    @property
    def allowed_difficulties(self) -> list[int]:
        return self.cfg.stage_difficulties[self.stage]

    @property
    def stage_name(self) -> str:
        return self.cfg.stage_names[self.stage]

    @property
    def rolling_accuracy(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history) / len(self.history)

    def update(self, correct: bool) -> None:
        """Record one prediction outcome."""
        self.history.append(1.0 if correct else 0.0)
        self._total_updates += 1

    def maybe_advance(self, global_step: int = 0) -> bool:
        """
        Check whether to advance to the next stage.
        Returns True if a stage transition occurred.
        """
        if self.stage >= len(self.cfg.stage_names) - 1:
            return False  # already at hardest stage

        if len(self.history) < self.cfg.min_window_before_advance:
            return False  # not enough data yet

        thresholds = [
            self.cfg.easy_to_medium_threshold,
            self.cfg.medium_to_hard_threshold,
        ]
        threshold = thresholds[min(self.stage, len(thresholds) - 1)]
        rolling_acc = self.rolling_accuracy

        if rolling_acc >= threshold:
            old_stage = self.stage
            self.stage += 1
            self.history.clear()  # reset window for next stage

            transition = {
                "step": global_step,
                "from_stage": self.cfg.stage_names[old_stage],
                "to_stage": self.stage_name,
                "rolling_acc_at_transition": rolling_acc,
                "total_examples_seen": self._total_updates,
            }
            self._stage_transitions.append(transition)

            logger.info(
                "[CurriculumGate] Step %d: advanced %s → %s "
                "(rolling acc=%.3f, threshold=%.2f)",
                global_step,
                transition["from_stage"],
                transition["to_stage"],
                rolling_acc,
                threshold,
            )

            # Log to W&B if available
            try:
                wandb.log({
                    "curriculum/stage": self.stage,
                    "curriculum/stage_name": self.stage_name,
                    "curriculum/rolling_acc_at_transition": rolling_acc,
                }, step=global_step)
            except Exception:
                pass  # W&B not initialised — skip silently

            return True

        return False

    def log_metrics(self, global_step: int) -> None:
        """Log current state to W&B."""
        try:
            wandb.log({
                "curriculum/stage": self.stage,
                "curriculum/rolling_acc": self.rolling_accuracy,
                "curriculum/window_size": len(self.history),
                "curriculum/allowed_difficulties": str(self.allowed_difficulties),
            }, step=global_step)
        except Exception:
            pass

    def summary(self) -> dict:
        """Return a summary dict for logging/reporting."""
        return {
            "final_stage": self.stage_name,
            "final_rolling_acc": self.rolling_accuracy,
            "total_examples_seen": self._total_updates,
            "stage_transitions": self._stage_transitions,
        }