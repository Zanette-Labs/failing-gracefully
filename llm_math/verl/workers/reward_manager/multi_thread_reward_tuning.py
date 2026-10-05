"""IDK rewards scaled by a per-c exponential moving average of accuracy."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Set

from verl.workers.reward_manager import register
from verl.workers.reward_manager.multi_thread_naive_idk import MultiThreadNaiveIdkRewardManager


@register("multi_thread_reward_tuning")
class MultiThreadRewardTuningManager(MultiThreadNaiveIdkRewardManager):
    """Score IDK as c * p_c, then update p_c from the completed training batch."""

    def __init__(self, *args, ema_alpha: float = 0.1, initial_success_rate: float = 0.0, **kwargs):
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1]")
        if not 0.0 <= initial_success_rate <= 1.0:
            raise ValueError("initial_success_rate must be in [0, 1]")
        self.ema_alpha = float(ema_alpha)
        self.initial_success_rate = float(initial_success_rate)
        self.success_ema: Dict[str, float] = {}
        self.update_counts: Dict[str, int] = {}
        self.update_ema = True
        super().__init__(*args, **kwargs)

    @staticmethod
    def _c_key(c: float) -> str:
        c = float(c)
        if not math.isfinite(c) or not 0.0 <= c <= 1.0:
            raise ValueError(f"idk_reward c must be finite and in [0, 1], got {c}")
        return format(c, ".12g")

    def set_update_ema(self, enabled: bool) -> None:
        self.update_ema = bool(enabled)

    def _effective_idk_reward(self, c: float) -> float:
        key = self._c_key(c)
        return c * self.success_ema.get(key, self.initial_success_rate)

    def _after_scoring(self, items, results, scored_indices: Set[int]) -> Dict[str, float]:
        # Exclude infrastructure failures: they are not observations of model success.
        counts = defaultdict(lambda: {"total": 0, "correct": 0, "idk": 0})
        present = {self._c_key(item["idk_reward"]) for item in items}
        for i in scored_indices:
            key = self._c_key(items[i]["idk_reward"])
            category = results[i]["category"]
            counts[key]["total"] += 1
            counts[key]["correct"] += category == "correct"
            counts[key]["idk"] += category == "idk"

        metrics: Dict[str, float] = {}
        for key in sorted(present, key=float):
            prior = self.success_ema.get(key, self.initial_success_rate)
            count = counts[key]
            total = count["total"]
            if total and self.update_ema:
                batch_success = count["correct"] / total
                self.success_ema[key] = (1.0 - self.ema_alpha) * prior + self.ema_alpha * batch_success
                self.update_counts[key] = self.update_counts.get(key, 0) + 1
            current = self.success_ema.get(key, self.initial_success_rate)
            prefix = f"reward_tuning/c_{key}"
            metrics[f"{prefix}/success_ema"] = current
            metrics[f"{prefix}/idk_reward_used"] = float(key) * prior
            metrics[f"{prefix}/batch_success"] = count["correct"] / total if total else 0.0
            metrics[f"{prefix}/batch_count"] = total
            metrics[f"{prefix}/batch_idk"] = count["idk"]
        return metrics

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ema_alpha": self.ema_alpha,
            "initial_success_rate": self.initial_success_rate,
            "success_ema": dict(self.success_ema),
            "update_counts": dict(self.update_counts),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state["ema_alpha"] != self.ema_alpha or state["initial_success_rate"] != self.initial_success_rate:
            raise ValueError("Reward tuning EMA settings differ from the checkpoint")
        self.success_ema = {self._c_key(float(c)): float(p) for c, p in state["success_ema"].items()}
        if any(not math.isfinite(p) or not 0.0 <= p <= 1.0 for p in self.success_ema.values()):
            raise ValueError("Checkpoint contains an invalid success EMA")
        self.update_counts = {self._c_key(float(c)): int(n) for c, n in state["update_counts"].items()}
