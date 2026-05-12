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
    """Actor using one shared backbone plus a main head and optional EA heads.

    TD3 calls ``forward(obs)`` and uses the main head. BA-UGD-ERL can call
    ``forward(obs, head_index=i)`` to evaluate an EA head that shares the same
    state representation but has separate head parameters.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray | Iterable[float],
        action_high: np.ndarray | Iterable[float],
        hidden_dim: int = 256,
        hidden_layers: int = 2,
        num_ea_heads: int = 0,
    ) -> None:
        super().__init__()
        if num_ea_heads < 0:
            raise ValueError("num_ea_heads must be >= 0")
        self.backbone = MLPBackbone(
            obs_dim=obs_dim,
            feature_dim=hidden_dim,
            hidden_dim=hidden_dim,
            hidden_layers=hidden_layers,
        )
        self.main_head = ActorHead(hidden_dim, action_dim)
        self.ea_heads = nn.ModuleList(
            ActorHead(hidden_dim, action_dim) for _ in range(num_ea_heads)
        )
        self.head = self.main_head

        low = torch.as_tensor(np.asarray(action_low), dtype=torch.float32)
        high = torch.as_tensor(np.asarray(action_high), dtype=torch.float32)
        scale = (high - low) / 2.0
        bias = (high + low) / 2.0
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_bias", bias)
        self.register_buffer("action_low", low)
        self.register_buffer("action_high", high)

    @property
    def num_ea_heads(self) -> int:
        return len(self.ea_heads)

    def _select_head(self, head_index: int | None) -> ActorHead:
        if head_index is None:
            return self.main_head
        if head_index < 0 or head_index >= self.num_ea_heads:
            raise IndexError(
                f"EA head index {head_index} out of range for {self.num_ea_heads} heads"
            )
        return self.ea_heads[head_index]

    def normalized_action(
        self, obs: torch.Tensor, head_index: int | None = None
    ) -> torch.Tensor:
        features = self.backbone(obs)
        return self._select_head(head_index)(features)

    def forward(self, obs: torch.Tensor, head_index: int | None = None) -> torch.Tensor:
        return self.normalized_action(head_index=head_index, obs=obs) * self.action_scale + self.action_bias


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
