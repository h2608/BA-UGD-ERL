from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Simple preallocated replay buffer for off-policy continuous control."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        capacity: int,
        device: torch.device,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.obs[self.ptr] = np.asarray(obs, dtype=np.float32)
        self.actions[self.ptr] = np.asarray(action, dtype=np.float32)
        self.rewards[self.ptr] = float(reward)
        self.next_obs[self.ptr] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if not self.can_sample(batch_size):
            raise ValueError(
                f"Cannot sample batch_size={batch_size} from buffer size={self.size}"
            )
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": torch.as_tensor(self.obs[idx], device=self.device),
            "actions": torch.as_tensor(self.actions[idx], device=self.device),
            "rewards": torch.as_tensor(self.rewards[idx], device=self.device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=self.device),
            "dones": torch.as_tensor(self.dones[idx], device=self.device),
        }


def _concat_batches(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batches[0].keys()
    return {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}


def sample_mixed_batch(
    rl_buffer: ReplayBuffer,
    pop_buffer: ReplayBuffer | None,
    batch_size: int,
    pop_fraction: float,
) -> dict[str, torch.Tensor] | None:
    """Sample a batch from B_rl and B_pop with deterministic fallbacks.

    If the combined buffers cannot provide a full batch without replacement,
    return None so the caller can skip the TD3 update.
    """

    pop_size = len(pop_buffer) if pop_buffer is not None else 0
    total_available = len(rl_buffer) + pop_size
    if total_available < batch_size:
        return None
    if pop_buffer is None or pop_size == 0:
        return rl_buffer.sample(batch_size) if rl_buffer.can_sample(batch_size) else None
    if len(rl_buffer) == 0:
        return pop_buffer.sample(batch_size) if pop_buffer.can_sample(batch_size) else None

    desired_pop = int(round(batch_size * float(np.clip(pop_fraction, 0.0, 1.0))))
    desired_rl = batch_size - desired_pop
    pop_count = min(desired_pop, pop_size)
    rl_count = min(desired_rl, len(rl_buffer))

    deficit = batch_size - pop_count - rl_count
    if deficit > 0:
        pop_extra = min(deficit, pop_size - pop_count)
        pop_count += pop_extra
        deficit -= pop_extra
    if deficit > 0:
        rl_extra = min(deficit, len(rl_buffer) - rl_count)
        rl_count += rl_extra
        deficit -= rl_extra
    if deficit > 0:
        return None

    batches: list[dict[str, torch.Tensor]] = []
    if rl_count > 0:
        batches.append(rl_buffer.sample(rl_count))
    if pop_count > 0:
        batches.append(pop_buffer.sample(pop_count))
    if not batches:
        return None
    if len(batches) == 1:
        return batches[0]
    return _concat_batches(batches)
