from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class TD3UpdateResult:
    critic_loss: float
    actor_loss: float | None


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(tau * source_param.data)


class TD3Updater:
    def __init__(
        self,
        actor: nn.Module,
        actor_target: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        gamma: float,
        tau: float,
        policy_noise: float,
        noise_clip: float,
        policy_delay: int,
    ) -> None:
        self.actor = actor
        self.actor_target = actor_target
        self.critic = critic
        self.critic_target = critic_target
        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = max(1, policy_delay)
        self.update_step = 0

    def update(self, batch: dict[str, torch.Tensor]) -> TD3UpdateResult:
        self.update_step += 1
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = self.actor_target(next_obs) + noise
            next_actions = torch.max(
                torch.min(next_actions, self.actor.action_high), self.actor.action_low
            )
            target_q1, target_q2 = self.critic_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target = rewards + (1.0 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(current_q1, target) + F.mse_loss(current_q2, target)

        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite critic loss detected")

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value: float | None = None
        if self.update_step % self.policy_delay == 0:
            actor_loss = -self.critic.q1_value(obs, self.actor(obs)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite actor loss detected")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()

            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)
            actor_loss_value = float(actor_loss.detach().cpu().item())

        return TD3UpdateResult(
            critic_loss=float(critic_loss.detach().cpu().item()),
            actor_loss=actor_loss_value,
        )
