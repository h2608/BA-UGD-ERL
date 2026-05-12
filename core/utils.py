from __future__ import annotations

import copy
import random
import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config at {path} did not parse to a dictionary")
    return config


def deep_update(base: dict[str, Any], updates: dict[str, Any] | None) -> dict[str, Any]:
    result = copy.deepcopy(base)
    if not updates:
        return result
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def save_config_copy(config_path: str | Path, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "config.yaml")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_env(env_name: str, seed: int | None = None) -> gym.Env:
    env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    obs, _ = env.reset(seed=seed)
    return np.asarray(obs, dtype=np.float32)


def step_env(
    env: gym.Env, action: np.ndarray
) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
    next_obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return np.asarray(next_obs, dtype=np.float32), float(reward), done, bool(truncated), info


def ensure_output_dirs(output_dir: str | Path, run_name: str) -> dict[str, Path]:
    root = Path(output_dir)
    run_dirs = {
        "root": root,
        "logs": root / "logs" / run_name,
        "models": root / "models" / run_name,
        "figures": root / "figures" / run_name,
    }
    for path in run_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return run_dirs


def save_checkpoint(
    path: str | Path,
    *,
    actor: torch.nn.Module,
    actor_target: torch.nn.Module,
    critic: torch.nn.Module,
    critic_target: torch.nn.Module,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    step: int,
) -> None:
    checkpoint = {
        "actor": actor.state_dict(),
        "actor_target": actor_target.state_dict(),
        "critic": critic.state_dict(),
        "critic_target": critic_target.state_dict(),
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "config": config,
        "step": step,
    }
    torch.save(checkpoint, path)


def as_float32_obs(obs: np.ndarray) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError("Observation contains NaN or Inf")
    return arr
