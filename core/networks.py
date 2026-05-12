from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from torch import nn


def build_mlp(
    input_dim: int,
    output_dim: int,
    hidden_dim: int = 256,
    hidden_layers: int = 2,
    activation: type[nn.Module] = nn.ReLU,
    output_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for _ in range(hidden_layers):
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(activation())
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class MLPBackbone(nn.Module):
    """State encoder shared by actor heads in later BA-UGD-ERL stages."""

    def __init__(
        self,
        obs_dim: int,
        feature_dim: int = 256,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be >= 1")
        layers: list[nn.Module] = []
        last_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        if last_dim != feature_dim:
            layers.append(nn.Linear(last_dim, feature_dim))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.output_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class ActorHead(nn.Module):
    """Actor head that maps shared features to tanh-normalized actions."""

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(feature_dim, action_dim), nn.Tanh())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class Actor(nn.Module):
    """TD3 actor using a backbone plus one head.

    The composition is intentionally compatible with future shared-backbone
    multi-head actors: BA-UGD-ERL can add more ActorHead modules without
    changing the main actor action scaling contract.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray | Iterable[float],
        action_high: np.ndarray | Iterable[float],
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.backbone = MLPBackbone(
            obs_dim=obs_dim,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.head = ActorHead(hidden_dim, action_dim)

        low = torch.as_tensor(np.asarray(action_low), dtype=torch.float32)
        high = torch.as_tensor(np.asarray(action_high), dtype=torch.float32)
        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    def normalized_action(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(obs)
        return self.head(features)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.normalized_action(obs) * self.action_scale + self.action_bias


class TwinCritic(nn.Module):
    """Twin Q networks used by TD3."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        input_dim = obs_dim + action_dim
        self.q1 = build_mlp(input_dim, 1, hidden_dim, hidden_layers)
        self.q2 = build_mlp(input_dim, 1, hidden_dim, hidden_layers)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)
