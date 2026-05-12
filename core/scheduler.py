from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MODE_TO_ID = {"Explore": 0, "Hybrid": 1, "Exploit": 2}


@dataclass
class ModeResources:
    active_ea_heads: int
    update_ratio: float
    pop_fraction: float


@dataclass
class SchedulerState:
    mode: str
    mode_id: int
    uncertainty_score: float
    critic_disagreement: float | None
    learning_progress: float | None
    progress_need: float


class UncertaintyScheduler:
    """EMA-smoothed three-state scheduler for BA-UGD-ERL."""

    def __init__(self, config: dict[str, Any], num_ea_heads: int) -> None:
        self.update_interval = int(config.get("update_interval", 5000))
        self.ema_alpha = float(config.get("ema_alpha", 0.8))
        self.disagreement_weight = float(config.get("disagreement_weight", 0.5))
        self.progress_weight = float(config.get("progress_weight", 0.5))
        self.progress_scale = float(config.get("progress_scale", 10.0))
        self.explore_enter = float(config.get("explore_enter", 0.65))
        self.explore_exit = float(config.get("explore_exit", 0.55))
        self.exploit_enter = float(config.get("exploit_enter", 0.35))
        self.exploit_exit = float(config.get("exploit_exit", 0.45))
        self.min_mode_steps = int(config.get("min_mode_steps", 5000))
        self.mode = str(config.get("initial_mode", "Hybrid"))
        if self.mode not in MODE_TO_ID:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")
        self.last_switch_step = 0
        self.ema_u: float | None = None
        self.ema_disagreement: float | None = None
        self.eval_returns: list[float] = []
        self.mode_resources = self._build_mode_resources(config, num_ea_heads)

    def _build_mode_resources(
        self, config: dict[str, Any], num_ea_heads: int
    ) -> dict[str, ModeResources]:
        defaults = {
            "Explore": {
                "active_ea_heads": num_ea_heads,
                "update_ratio": 0.5,
                "pop_fraction": 0.7,
            },
            "Hybrid": {
                "active_ea_heads": max(1, min(2, num_ea_heads)),
                "update_ratio": 1.0,
                "pop_fraction": 0.5,
            },
            "Exploit": {
                "active_ea_heads": 1 if num_ea_heads > 0 else 0,
                "update_ratio": 1.0,
                "pop_fraction": 0.2,
            },
        }
        configured = config.get("modes", {})
        resources: dict[str, ModeResources] = {}
        for mode, default in defaults.items():
            values = {**default, **configured.get(mode, {})}
            resources[mode] = ModeResources(
                active_ea_heads=max(
                    0, min(int(values["active_ea_heads"]), num_ea_heads)
                ),
                update_ratio=float(values["update_ratio"]),
                pop_fraction=float(np.clip(values["pop_fraction"], 0.0, 1.0)),
            )
        return resources

    def record_eval_return(self, eval_return: float | None) -> None:
        if eval_return is not None and np.isfinite(eval_return):
            self.eval_returns.append(float(eval_return))

    def current_resources(self) -> ModeResources:
        return self.mode_resources[self.mode]

    def _learning_progress(self) -> float | None:
        if len(self.eval_returns) < 2:
            return None
        return self.eval_returns[-1] - self.eval_returns[-2]

    def update(
        self, step: int, critic_disagreement: float | None
    ) -> SchedulerState:
        if critic_disagreement is None or not np.isfinite(critic_disagreement):
            disagreement_score = 0.5
            disagreement_value = None
        else:
            disagreement_value = float(critic_disagreement)
            if self.ema_disagreement is None:
                self.ema_disagreement = max(disagreement_value, 1e-6)
            else:
                self.ema_disagreement = (
                    self.ema_alpha * self.ema_disagreement
                    + (1.0 - self.ema_alpha) * disagreement_value
                )
            disagreement_score = float(
                np.clip(disagreement_value / (self.ema_disagreement + 1e-6), 0.0, 2.0)
                / 2.0
            )

        progress = self._learning_progress()
        if progress is None:
            progress_need = 0.5
        else:
            normalized_progress = np.clip(progress / self.progress_scale, 0.0, 1.0)
            progress_need = float(1.0 - normalized_progress)

        raw_u = (
            self.disagreement_weight * disagreement_score
            + self.progress_weight * progress_need
        )
        if self.ema_u is None:
            self.ema_u = raw_u
        else:
            self.ema_u = self.ema_alpha * self.ema_u + (1.0 - self.ema_alpha) * raw_u

        can_switch = (step - self.last_switch_step) >= self.min_mode_steps
        if can_switch:
            previous_mode = self.mode
            if self.mode == "Explore":
                if self.ema_u < self.explore_exit:
                    self.mode = "Hybrid"
            elif self.mode == "Exploit":
                if self.ema_u > self.exploit_exit:
                    self.mode = "Hybrid"
            else:
                if self.ema_u >= self.explore_enter:
                    self.mode = "Explore"
                elif self.ema_u <= self.exploit_enter:
                    self.mode = "Exploit"
            if self.mode != previous_mode:
                self.last_switch_step = step

        return SchedulerState(
            mode=self.mode,
            mode_id=MODE_TO_ID[self.mode],
            uncertainty_score=float(self.ema_u),
            critic_disagreement=disagreement_value,
            learning_progress=progress,
            progress_need=progress_need,
        )
