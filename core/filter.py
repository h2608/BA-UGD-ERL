from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class FilterDecision:
    accepted: bool
    threshold: float | None
    rolling_best: float | None
    reason: str


class TrajectoryFilter:
    """Trajectory-level return filter using one rolling-best threshold rule."""

    def __init__(self, warmup_episodes: int = 5, return_margin: float = 100.0) -> None:
        self.warmup_episodes = max(0, int(warmup_episodes))
        self.return_margin = float(return_margin)
        self.observed = 0
        self.accepted = 0
        self.rolling_best: float | None = None
        self.last_threshold: float | None = None

    @property
    def acceptance_rate(self) -> float:
        if self.observed == 0:
            return 0.0
        return self.accepted / self.observed

    def evaluate(self, episode_return: float) -> FilterDecision:
        self.observed += 1
        if not math.isfinite(episode_return):
            return FilterDecision(
                accepted=False,
                threshold=self.last_threshold,
                rolling_best=self.rolling_best,
                reason="non_finite_return",
            )

        if self.rolling_best is None:
            self.rolling_best = episode_return

        if self.observed <= self.warmup_episodes:
            self.accepted += 1
            self.rolling_best = max(self.rolling_best, episode_return)
            self.last_threshold = None
            return FilterDecision(
                accepted=True,
                threshold=None,
                rolling_best=self.rolling_best,
                reason="warmup",
            )

        threshold = self.rolling_best - self.return_margin
        self.last_threshold = threshold
        accepted = episode_return >= threshold
        if accepted:
            self.accepted += 1
        self.rolling_best = max(self.rolling_best, episode_return)
        return FilterDecision(
            accepted=accepted,
            threshold=threshold,
            rolling_best=self.rolling_best,
            reason="threshold",
        )
