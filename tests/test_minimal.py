from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.networks import Actor, TwinCritic
from core.replay_buffer import ReplayBuffer
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
    )
    critic = TwinCritic(obs_dim=11, action_dim=3, hidden_dim=64, hidden_layers=2)
    obs = torch.zeros(4, 11)
    action = actor(obs)
    q1, q2 = critic(obs, action)
    assert action.shape == (4, 3)
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


if __name__ == "__main__":
    test_config_loads()
    test_network_shapes()
    test_replay_buffer_sample()
    print("minimal tests passed")
