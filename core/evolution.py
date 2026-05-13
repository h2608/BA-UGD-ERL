from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from core.networks import Actor, ActorHead


@dataclass
class EvolutionResult:
    evolved: bool
    elite_head: int | None
    elite_fitness: float | None
    mutated_heads: list[int]
    reason: str


class HeadEvolution:
    """Lightweight EA update over actor heads only."""

    def __init__(self, mutation_std: float = 0.05) -> None:
        if mutation_std < 0:
            raise ValueError("mutation_std must be >= 0")
        self.mutation_std = float(mutation_std)

    def evolve(self, actor: Actor, ea_results: list[dict[str, Any]]) -> EvolutionResult:
        fitness_by_head: dict[int, list[float]] = {}
        for item in ea_results:
            head_index = int(item["head_index"])
            if 0 <= head_index < actor.num_ea_heads:
                fitness_by_head.setdefault(head_index, []).append(
                    float(item["episode_return"])
                )

        if len(fitness_by_head) < 2:
            return EvolutionResult(False, None, None, [], "need_at_least_two_heads")

        mean_fitness = {
            head_index: sum(values) / len(values)
            for head_index, values in fitness_by_head.items()
        }
        ranked = sorted(mean_fitness.items(), key=lambda item: item[1], reverse=True)
        elite_head = ranked[0][0]
        elite_fitness = ranked[0][1]
        elite_state = {
            key: value.detach().clone()
            for key, value in actor.ea_heads[elite_head].state_dict().items()
        }

        mutated_heads: list[int] = []
        with torch.no_grad():
            for head_index, _ in ranked[1:]:
                head = actor.ea_heads[head_index]
                head.load_state_dict(elite_state)
                self._mutate_head(head)
                mutated_heads.append(head_index)

        return EvolutionResult(
            evolved=True,
            elite_head=elite_head,
            elite_fitness=elite_fitness,
            mutated_heads=mutated_heads,
            reason="elite_clone_mutation",
        )

    def _mutate_head(self, head: ActorHead) -> None:
        if self.mutation_std == 0:
            return
        for parameter in head.parameters():
            parameter.add_(torch.randn_like(parameter) * self.mutation_std)
