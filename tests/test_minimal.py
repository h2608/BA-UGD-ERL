from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.networks import Actor, TwinCritic
from core.filter import TrajectoryFilter
from core.replay_buffer import ReplayBuffer, sample_mixed_batch
from core.scheduler import UncertaintyScheduler
from core.utils import load_config


def test_config_loads() -> None:
    config = load_config(ROOT / "configs" / "hopper.yaml")
    assert config["experiment"]["algorithm"] == "td3_only"
    assert config["td3"]["warmup_steps"] >= 5000


def test_network_shapes() -> None:
    actor = Actor(
        obs_dim=11,
        action_dim=3,
        action_low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        hidden_dim=64,
        hidden_layers=2,
        num_ea_heads=2,
    )
    critic = TwinCritic(obs_dim=11, action_dim=3, hidden_dim=64, hidden_layers=2)
    obs = torch.zeros(4, 11)
    action = actor(obs)
    ea_action = actor(obs, head_index=1)
    q1, q2 = critic(obs, action)
    assert action.shape == (4, 3)
    assert ea_action.shape == (4, 3)
    assert actor.num_ea_heads == 2
    assert q1.shape == (4, 1)
    assert q2.shape == (4, 1)


def test_replay_buffer_sample() -> None:
    buffer = ReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"))
    for idx in range(6):
        obs = np.array([idx, idx + 1], dtype=np.float32)
        action = np.array([0.1], dtype=np.float32)
        buffer.add(obs, action, 1.0, obs + 1.0, False)
    batch = buffer.sample(4)
    assert batch["obs"].shape == (4, 2)
    assert batch["actions"].shape == (4, 1)
    assert batch["rewards"].shape == (4, 1)


def test_mixed_sampling_fallback() -> None:
    rl_buffer = ReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"))
    pop_buffer = ReplayBuffer(obs_dim=2, action_dim=1, capacity=8, device=torch.device("cpu"))
    for idx in range(4):
        obs = np.array([idx, idx + 1], dtype=np.float32)
        action = np.array([0.1], dtype=np.float32)
        rl_buffer.add(obs, action, 1.0, obs + 1.0, False)
    for idx in range(2):
        obs = np.array([idx, idx + 1], dtype=np.float32)
        action = np.array([0.2], dtype=np.float32)
        pop_buffer.add(obs, action, 2.0, obs + 1.0, False)
    assert sample_mixed_batch(rl_buffer, pop_buffer, batch_size=8, pop_fraction=0.5) is None
    batch = sample_mixed_batch(rl_buffer, pop_buffer, batch_size=4, pop_fraction=0.75)
    assert batch is not None
    assert batch["obs"].shape == (4, 2)


def test_trajectory_filter() -> None:
    trajectory_filter = TrajectoryFilter(warmup_episodes=1, return_margin=10.0)
    assert trajectory_filter.evaluate(100.0).accepted
    assert trajectory_filter.evaluate(95.0).accepted
    assert not trajectory_filter.evaluate(80.0).accepted
    assert trajectory_filter.acceptance_rate == 2 / 3


def test_scheduler_outputs_mode() -> None:
    scheduler = UncertaintyScheduler(
        {
            "initial_mode": "Hybrid",
            "min_mode_steps": 0,
            "modes": {
                "Explore": {"active_ea_heads": 4, "update_ratio": 0.5, "pop_fraction": 0.7},
                "Hybrid": {"active_ea_heads": 2, "update_ratio": 1.0, "pop_fraction": 0.5},
                "Exploit": {"active_ea_heads": 1, "update_ratio": 1.0, "pop_fraction": 0.2},
            },
        },
        num_ea_heads=4,
    )
    scheduler.record_eval_return(10.0)
    scheduler.record_eval_return(10.0)
    state = scheduler.update(step=1, critic_disagreement=1.0)
    assert state.mode in {"Explore", "Hybrid", "Exploit"}
    assert scheduler.current_resources().active_ea_heads >= 1


if __name__ == "__main__":
    test_config_loads()
    test_network_shapes()
    test_replay_buffer_sample()
    test_mixed_sampling_fallback()
    test_trajectory_filter()
    test_scheduler_outputs_mode()
    print("minimal tests passed")
