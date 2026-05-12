from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from core.logger import ExperimentLogger
from core.networks import Actor, TwinCritic
from core.replay_buffer import ReplayBuffer
from core.td3_update import TD3Updater
from core.utils import (
    as_float32_obs,
    deep_update,
    ensure_output_dirs,
    get_device,
    make_env,
    reset_env,
    save_checkpoint,
    set_seed,
)


def _require_box_spaces(env: gym.Env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise TypeError("Only Box observation spaces are supported in this prototype")
    if not isinstance(env.action_space, gym.spaces.Box):
        raise TypeError("Only Box action spaces are supported in this prototype")


def build_td3_components(
    config: dict[str, Any],
    env: gym.Env,
    device: torch.device,
) -> dict[str, Any]:
    _require_box_spaces(env)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    network_cfg = config["network"]
    td3_cfg = config["td3"]

    actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=int(network_cfg["hidden_dim"]),
        hidden_layers=int(network_cfg["hidden_layers"]),
    ).to(device)
    actor_target = copy.deepcopy(actor).to(device)

    critic = TwinCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=int(network_cfg["hidden_dim"]),
        hidden_layers=int(network_cfg["hidden_layers"]),
    ).to(device)
    critic_target = copy.deepcopy(critic).to(device)

    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=float(td3_cfg["actor_lr"]))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=float(td3_cfg["critic_lr"]))
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim,
        action_dim=action_dim,
        capacity=int(td3_cfg["buffer_size"]),
        device=device,
    )
    updater = TD3Updater(
        actor=actor,
        actor_target=actor_target,
        critic=critic,
        critic_target=critic_target,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        gamma=float(td3_cfg["gamma"]),
        tau=float(td3_cfg["tau"]),
        policy_noise=float(td3_cfg["policy_noise"]),
        noise_clip=float(td3_cfg["noise_clip"]),
        policy_delay=int(td3_cfg["policy_delay"]),
    )
    return {
        "actor": actor,
        "actor_target": actor_target,
        "critic": critic,
        "critic_target": critic_target,
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "replay_buffer": replay_buffer,
        "updater": updater,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }


@torch.no_grad()
def select_action(actor: Actor, obs: np.ndarray, device: torch.device) -> np.ndarray:
    actor.eval()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action = actor(obs_tensor).squeeze(0).cpu().numpy()
    actor.train()
    return action


def add_exploration_noise(
    action: np.ndarray,
    action_space: gym.spaces.Box,
    noise_std: float,
) -> np.ndarray:
    scale = (action_space.high - action_space.low) / 2.0
    noise = np.random.normal(0.0, noise_std * scale, size=action_space.shape)
    noisy_action = action + noise
    return np.clip(noisy_action, action_space.low, action_space.high).astype(np.float32)


@torch.no_grad()
def evaluate_actor(
    actor: Actor,
    env_name: str,
    seed: int,
    episodes: int,
    device: torch.device,
) -> float:
    eval_env = make_env(env_name, seed=seed)
    returns: list[float] = []
    try:
        for episode_idx in range(episodes):
            obs = reset_env(eval_env, seed=seed + episode_idx)
            done = False
            episode_return = 0.0
            while not done:
                action = select_action(actor, obs, device)
                next_obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = bool(terminated or truncated)
                obs = as_float32_obs(next_obs)
                episode_return += float(reward)
            returns.append(episode_return)
    finally:
        eval_env.close()
    return float(np.mean(returns)) if returns else float("nan")


def _write_effective_config(config: dict[str, Any], run_dir: Path) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def run_env_smoke(config: dict[str, Any]) -> dict[str, Any]:
    config = deep_update(config, None)
    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = get_device()
    env = make_env(config["env"]["name"], seed=seed)
    try:
        _require_box_spaces(env)
        obs = reset_env(env, seed=seed)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        components = build_td3_components(config, env, device)
        actor = components["actor"]
        test_action = select_action(actor, obs, device)
        return {
            "env_name": config["env"]["name"],
            "obs_shape": tuple(np.asarray(obs).shape),
            "next_obs_shape": tuple(np.asarray(next_obs).shape),
            "action_shape": tuple(np.asarray(test_action).shape),
            "reward": float(reward),
            "done": bool(terminated or truncated),
            "device": str(device),
        }
    finally:
        env.close()


def run_training(
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = deep_update(config, overrides)
    if config["experiment"].get("algorithm") != "td3_only":
        raise NotImplementedError("Stage A/B only implements algorithm: td3_only")

    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = get_device()
    total_steps = int(config["experiment"]["total_steps"])
    td3_cfg = config["td3"]
    logging_cfg = config["logging"]
    warmup_steps = max(5000, int(td3_cfg["warmup_steps"]))
    batch_size = int(td3_cfg["batch_size"])
    update_ratio = float(td3_cfg.get("update_ratio", 1.0))
    eval_interval = int(logging_cfg["eval_interval"])
    checkpoint_interval = int(logging_cfg["checkpoint_interval"])

    run_name = f"{config['experiment']['name']}_seed{seed}_{int(time.time())}"
    dirs = ensure_output_dirs(config["experiment"].get("output_dir", "outputs"), run_name)
    _write_effective_config(config, dirs["logs"])

    env = make_env(config["env"]["name"], seed=seed)
    logger = ExperimentLogger(
        dirs["logs"], console_interval=int(logging_cfg.get("console_interval", 1000))
    )
    last_critic_loss: float | None = None
    last_actor_loss: float | None = None
    last_eval_return: float | None = None
    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    update_accumulator = 0.0
    update_count = 0

    try:
        components = build_td3_components(config, env, device)
        actor: Actor = components["actor"]
        actor_target: Actor = components["actor_target"]
        critic = components["critic"]
        critic_target = components["critic_target"]
        actor_optimizer = components["actor_optimizer"]
        critic_optimizer = components["critic_optimizer"]
        replay_buffer: ReplayBuffer = components["replay_buffer"]
        updater: TD3Updater = components["updater"]

        obs = reset_env(env, seed=seed)
        for step in range(1, total_steps + 1):
            if step <= warmup_steps:
                action = env.action_space.sample().astype(np.float32)
            else:
                action = select_action(actor, obs, device)
                action = add_exploration_noise(
                    action,
                    env.action_space,
                    noise_std=float(td3_cfg["exploration_noise"]),
                )

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            done_for_bootstrap = bool(terminated)
            next_obs = as_float32_obs(next_obs)
            replay_buffer.add(obs, action, float(reward), next_obs, done_for_bootstrap)

            episode_return += float(reward)
            episode_len += 1
            obs = next_obs

            if done:
                episode_idx += 1
                logger.scalar("train/episode_return", episode_return, step)
                logger.scalar("train/episode_length", episode_len, step)
                obs = reset_env(env)
                episode_return = 0.0
                episode_len = 0

            if step > warmup_steps and replay_buffer.can_sample(batch_size):
                update_accumulator += update_ratio
                while update_accumulator >= 1.0:
                    batch = replay_buffer.sample(batch_size)
                    update_result = updater.update(batch)
                    last_critic_loss = update_result.critic_loss
                    last_actor_loss = update_result.actor_loss
                    update_count += 1
                    update_accumulator -= 1.0

            scalar_values = {
                "train/replay_buffer_size": len(replay_buffer),
                "train/update_count": update_count,
            }
            if last_critic_loss is not None:
                scalar_values["loss/critic"] = last_critic_loss
            if last_actor_loss is not None:
                scalar_values["loss/actor"] = last_actor_loss
            logger.scalars(scalar_values, step)

            if step % eval_interval == 0 or step == total_steps:
                last_eval_return = evaluate_actor(
                    actor=actor,
                    env_name=config["env"]["name"],
                    seed=seed + 10000 + step,
                    episodes=int(config["env"].get("eval_episodes", 5)),
                    device=device,
                )
                logger.scalar("eval/return", last_eval_return, step)

            if step % checkpoint_interval == 0 or step == total_steps:
                checkpoint_path = dirs["models"] / f"step_{step}.pt"
                save_checkpoint(
                    checkpoint_path,
                    actor=actor,
                    actor_target=actor_target,
                    critic=critic,
                    critic_target=critic_target,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    config=config,
                    step=step,
                )
                latest_path = dirs["models"] / "latest.pt"
                save_checkpoint(
                    latest_path,
                    actor=actor,
                    actor_target=actor_target,
                    critic=critic,
                    critic_target=critic_target,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    config=config,
                    step=step,
                )
                root_latest = dirs["root"] / "models" / "latest.pt"
                save_checkpoint(
                    root_latest,
                    actor=actor,
                    actor_target=actor_target,
                    critic=critic,
                    critic_target=critic_target,
                    actor_optimizer=actor_optimizer,
                    critic_optimizer=critic_optimizer,
                    config=config,
                    step=step,
                )

            logger.maybe_console(
                step,
                total_steps,
                {
                    "episodes": episode_idx,
                    "return": episode_return if episode_len else None,
                    "eval": last_eval_return,
                    "critic_loss": last_critic_loss,
                    "actor_loss": last_actor_loss,
                    "B_rl": len(replay_buffer),
                    "updates": update_count,
                },
            )

        return {
            "run_name": run_name,
            "log_dir": str(dirs["logs"]),
            "model_dir": str(dirs["models"]),
            "latest_checkpoint": str(dirs["models"] / "latest.pt"),
            "root_latest_checkpoint": str(dirs["root"] / "models" / "latest.pt"),
            "total_steps": total_steps,
            "updates": update_count,
            "episodes": episode_idx,
            "last_eval_return": last_eval_return,
            "device": str(device),
        }
    finally:
        logger.close()
        env.close()
