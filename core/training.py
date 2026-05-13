from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml

from core.evolution import EvolutionResult, HeadEvolution
from core.logger import ExperimentLogger
from core.filter import TrajectoryFilter
from core.networks import Actor, TwinCritic
from core.replay_buffer import ReplayBuffer, sample_mixed_batch
from core.scheduler import (
    MODE_TO_ID,
    SchedulerState,
    StaticSwitchScheduler,
    UncertaintyScheduler,
)
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
    ba_cfg = config.get("ba_ugd_erl", {})
    ba_enabled = bool(ba_cfg.get("enabled", False))
    num_ea_heads = int(ba_cfg.get("num_ea_heads", 0)) if ba_enabled else 0

    actor = Actor(
        obs_dim=obs_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        hidden_dim=int(network_cfg["hidden_dim"]),
        hidden_layers=int(network_cfg["hidden_layers"]),
        num_ea_heads=num_ea_heads,
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
    pop_buffer = None
    if ba_enabled and bool(ba_cfg.get("mixed_sampling", {}).get("enabled", False)):
        pop_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            action_dim=action_dim,
            capacity=int(ba_cfg.get("pop_buffer_size", td3_cfg["buffer_size"])),
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
        "pop_buffer": pop_buffer,
        "updater": updater,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
    }


@torch.no_grad()
def select_action(
    actor: Actor,
    obs: np.ndarray,
    device: torch.device,
    head_index: int | None = None,
) -> np.ndarray:
    actor.eval()
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    action = actor(obs_tensor, head_index=head_index).squeeze(0).cpu().numpy()
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


@torch.no_grad()
def compute_critic_disagreement(
    actor: Actor,
    critic: TwinCritic,
    replay_buffer: ReplayBuffer,
    batch_size: int,
) -> float | None:
    if not replay_buffer.can_sample(batch_size):
        return None
    batch = replay_buffer.sample(batch_size)
    actions = actor(batch["obs"])
    q1, q2 = critic(batch["obs"], actions)
    disagreement = torch.mean(torch.abs(q1 - q2))
    if not torch.isfinite(disagreement):
        return None
    return float(disagreement.detach().cpu().item())


def rollout_ea_heads(
    actor: Actor,
    env_name: str,
    seed: int,
    active_heads: int,
    episodes_per_head: int,
    max_rollout_steps: int,
    noise_std: float,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Roll out EA heads for diagnostics.

    Stage C step 1 only evaluates EA heads. It does not evolve head
    parameters and does not store the generated transitions.
    """

    if actor.num_ea_heads == 0:
        return []
    rollout_count = min(active_heads, actor.num_ea_heads)
    results: list[dict[str, Any]] = []
    for head_index in range(rollout_count):
        for episode_idx in range(episodes_per_head):
            env = make_env(env_name, seed=seed + head_index * 1000 + episode_idx)
            try:
                obs = reset_env(env, seed=seed + head_index * 1000 + episode_idx)
                done = False
                episode_return = 0.0
                episode_len = 0
                transitions: list[
                    tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]
                ] = []
                while not done and episode_len < max_rollout_steps:
                    action = select_action(actor, obs, device, head_index=head_index)
                    action = add_exploration_noise(action, env.action_space, noise_std)
                    transition_obs = obs.copy()
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    done = bool(terminated or truncated)
                    done_for_bootstrap = bool(terminated)
                    next_obs = as_float32_obs(next_obs)
                    episode_return += float(reward)
                    episode_len += 1
                    transitions.append(
                        (
                            transition_obs,
                            action.copy(),
                            float(reward),
                            next_obs.copy(),
                            done_for_bootstrap,
                        )
                    )
                    obs = next_obs
                results.append(
                    {
                        "head_index": head_index,
                        "episode_return": episode_return,
                        "episode_length": episode_len,
                        "transitions": transitions,
                    }
                )
            finally:
                env.close()
    return results


def _write_effective_config(config: dict[str, Any], run_dir: Path) -> None:
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=_json_default, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, default=_json_default, ensure_ascii=False, indent=2)


def _print_config_summary(config: dict[str, Any], device: torch.device) -> None:
    experiment_cfg = config["experiment"]
    td3_cfg = config["td3"]
    ba_cfg = config.get("ba_ugd_erl", {})
    summary = {
        "algorithm": experiment_cfg.get("algorithm"),
        "env": config["env"]["name"],
        "seed": experiment_cfg.get("seed"),
        "total_steps": experiment_cfg.get("total_steps"),
        "warmup_steps": max(5000, int(td3_cfg["warmup_steps"])),
        "batch_size": td3_cfg["batch_size"],
        "update_ratio": td3_cfg.get("update_ratio", 1.0),
        "device": str(device),
        "ba_enabled": bool(ba_cfg.get("enabled", False)),
        "ea_heads": int(ba_cfg.get("num_ea_heads", 0))
        if bool(ba_cfg.get("enabled", False))
        else 0,
        "mixed_sampling": bool(
            ba_cfg.get("mixed_sampling", {}).get("enabled", False)
        ),
        "trajectory_filter": bool(ba_cfg.get("filter", {}).get("enabled", False)),
        "evolution": bool(ba_cfg.get("evolution", {}).get("enabled", False)),
        "scheduler": bool(config.get("scheduler", {}).get("enabled", False)),
        "scheduler_strategy": config.get("scheduler", {}).get("strategy", "none"),
    }
    print("Training config summary:", flush=True)
    for key, value in summary.items():
        print(f"  {key}: {value}", flush=True)


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
    algorithm = config["experiment"].get("algorithm")
    if algorithm not in {"td3_only", "ba_ugd_erl"}:
        raise NotImplementedError(f"Unsupported algorithm: {algorithm}")
    ba_enabled = algorithm == "ba_ugd_erl" and bool(
        config.get("ba_ugd_erl", {}).get("enabled", False)
    )

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
    ba_cfg = config.get("ba_ugd_erl", {})
    ea_cfg = ba_cfg.get("ea", {})
    ea_rollout_enabled = ba_enabled and bool(ea_cfg.get("rollout_enabled", False))
    ea_rollout_interval = int(ea_cfg.get("rollout_interval", 5000))
    ea_active_heads = int(ea_cfg.get("active_heads", ba_cfg.get("num_ea_heads", 0)))
    ea_episodes_per_head = int(ea_cfg.get("rollout_episodes_per_head", 1))
    ea_max_rollout_steps = int(ea_cfg.get("max_rollout_steps", 1000))
    ea_noise_std = float(ea_cfg.get("exploration_noise", 0.1))
    mixed_cfg = ba_cfg.get("mixed_sampling", {})
    mixed_sampling_enabled = ba_enabled and bool(mixed_cfg.get("enabled", False))
    pop_fraction = float(mixed_cfg.get("pop_fraction", 0.5))
    filter_cfg = ba_cfg.get("filter", {})
    filter_enabled = ba_enabled and bool(filter_cfg.get("enabled", False))
    evolution_cfg = ba_cfg.get("evolution", {})
    evolution_enabled = ba_enabled and bool(evolution_cfg.get("enabled", False))
    evolution_interval = int(evolution_cfg.get("interval", ea_rollout_interval))
    scheduler_cfg = config.get("scheduler", {})
    scheduler_enabled = ba_enabled and bool(scheduler_cfg.get("enabled", False))
    scheduler_strategy = str(scheduler_cfg.get("strategy", "uncertainty"))
    scheduler_update_interval = int(scheduler_cfg.get("update_interval", 5000))

    run_name = f"{config['experiment']['name']}_seed{seed}_{int(time.time())}"
    dirs = ensure_output_dirs(config["experiment"].get("output_dir", "outputs"), run_name)
    _write_effective_config(config, dirs["logs"])
    _print_config_summary(config, device)

    env = make_env(config["env"]["name"], seed=seed)
    wall_clock_start = time.perf_counter()
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
    last_ea_mean_return: float | None = None
    last_ea_active_heads = 0
    last_filter_acceptance_rate: float | None = None
    last_evolution_result: EvolutionResult | None = None
    last_scheduler_state: SchedulerState | None = None
    current_mode = "TD3" if not ba_enabled else "Hybrid"
    current_mode_id = MODE_TO_ID.get(current_mode, -1)
    current_update_ratio = update_ratio
    current_pop_fraction = pop_fraction
    current_active_heads = ea_active_heads
    mode_step_counts: dict[str, int] = {}
    scheduler_trace: list[dict[str, Any]] = []
    mode_switches: list[dict[str, Any]] = []
    trajectory_filter = (
        TrajectoryFilter(
            warmup_episodes=int(filter_cfg.get("warmup_episodes", 5)),
            return_margin=float(filter_cfg.get("return_margin", 100.0)),
        )
        if filter_enabled
        else None
    )
    scheduler = None
    if scheduler_enabled:
        if scheduler_strategy == "static_switch":
            scheduler = StaticSwitchScheduler(
                config=scheduler_cfg,
                num_ea_heads=int(ba_cfg.get("num_ea_heads", 0)),
                total_steps=total_steps,
            )
        elif scheduler_strategy == "uncertainty":
            scheduler = UncertaintyScheduler(
                config=scheduler_cfg,
                num_ea_heads=int(ba_cfg.get("num_ea_heads", 0)),
            )
        else:
            raise ValueError(f"Unknown scheduler strategy: {scheduler_strategy}")
    evolver = (
        HeadEvolution(mutation_std=float(evolution_cfg.get("mutation_std", 0.05)))
        if evolution_enabled
        else None
    )
    if scheduler is not None:
        resources = scheduler.current_resources()
        current_mode = scheduler.mode
        current_mode_id = MODE_TO_ID[current_mode]
        current_update_ratio = resources.update_ratio
        current_pop_fraction = resources.pop_fraction
        current_active_heads = resources.active_ea_heads

    try:
        components = build_td3_components(config, env, device)
        actor: Actor = components["actor"]
        actor_target: Actor = components["actor_target"]
        critic = components["critic"]
        critic_target = components["critic_target"]
        actor_optimizer = components["actor_optimizer"]
        critic_optimizer = components["critic_optimizer"]
        replay_buffer: ReplayBuffer = components["replay_buffer"]
        pop_buffer: ReplayBuffer | None = components["pop_buffer"]
        updater: TD3Updater = components["updater"]

        obs = reset_env(env, seed=seed)
        for step in range(1, total_steps + 1):
            if ba_enabled:
                mode_step_counts[current_mode] = mode_step_counts.get(current_mode, 0) + 1

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

            if step > warmup_steps:
                update_accumulator += current_update_ratio
                while update_accumulator >= 1.0:
                    if mixed_sampling_enabled:
                        batch = sample_mixed_batch(
                            replay_buffer,
                            pop_buffer,
                            batch_size=batch_size,
                            pop_fraction=current_pop_fraction,
                        )
                    else:
                        batch = (
                            replay_buffer.sample(batch_size)
                            if replay_buffer.can_sample(batch_size)
                            else None
                        )
                    if batch is None:
                        break
                    update_result = updater.update(batch)
                    last_critic_loss = update_result.critic_loss
                    last_actor_loss = update_result.actor_loss
                    update_count += 1
                    update_accumulator -= 1.0

            if ea_rollout_enabled and step % ea_rollout_interval == 0:
                ea_results = rollout_ea_heads(
                    actor=actor,
                    env_name=config["env"]["name"],
                    seed=seed + 20000 + step,
                    active_heads=current_active_heads,
                    episodes_per_head=ea_episodes_per_head,
                    max_rollout_steps=ea_max_rollout_steps,
                    noise_std=ea_noise_std,
                    device=device,
                )
                if ea_results:
                    returns = [item["episode_return"] for item in ea_results]
                    last_ea_mean_return = float(np.mean(returns))
                    last_ea_active_heads = len({item["head_index"] for item in ea_results})
                    stored_transitions = 0
                    accepted_trajectories = 0
                    last_threshold: float | None = None
                    rolling_best: float | None = None
                    if pop_buffer is not None:
                        for item in ea_results:
                            accepted = True
                            if trajectory_filter is not None:
                                decision = trajectory_filter.evaluate(
                                    float(item["episode_return"])
                                )
                                accepted = decision.accepted
                                last_threshold = decision.threshold
                                rolling_best = decision.rolling_best
                            if accepted:
                                accepted_trajectories += 1
                                for transition in item["transitions"]:
                                    pop_buffer.add(*transition)
                                    stored_transitions += 1
                    if trajectory_filter is None:
                        accepted_trajectories = len(ea_results)
                        last_filter_acceptance_rate = 1.0
                    else:
                        last_filter_acceptance_rate = trajectory_filter.acceptance_rate
                    logger.scalar("ea/mean_return", last_ea_mean_return, step)
                    logger.scalar("ea/active_heads", last_ea_active_heads, step)
                    logger.scalar("ea/stored_transitions", stored_transitions, step)
                    logger.scalar("filter/accepted_trajectories", accepted_trajectories, step)
                    logger.scalar("filter/acceptance_rate", last_filter_acceptance_rate, step)
                    logger.scalar("filter/threshold", last_threshold, step)
                    logger.scalar("filter/rolling_best", rolling_best, step)
                    for item in ea_results:
                        logger.scalar(
                            f"ea/head_{item['head_index']}_return",
                            item["episode_return"],
                            step,
                        )
                    if evolver is not None and step % evolution_interval == 0:
                        last_evolution_result = evolver.evolve(actor, ea_results)
                        if last_evolution_result.evolved:
                            logger.scalar(
                                "evolution/elite_head",
                                last_evolution_result.elite_head,
                                step,
                            )
                            logger.scalar(
                                "evolution/elite_fitness",
                                last_evolution_result.elite_fitness,
                                step,
                            )
                            logger.scalar(
                                "evolution/mutated_heads",
                                len(last_evolution_result.mutated_heads),
                                step,
                            )

            scalar_values = {
                "train/B_rl_size": len(replay_buffer),
                "train/update_count": update_count,
                "train/current_update_ratio": current_update_ratio,
            }
            if pop_buffer is not None:
                scalar_values["train/B_pop_size"] = len(pop_buffer)
            if last_filter_acceptance_rate is not None:
                scalar_values["filter/acceptance_rate"] = last_filter_acceptance_rate
            if ba_enabled:
                scalar_values["scheduler/mode_id"] = current_mode_id
            if last_evolution_result is not None and last_evolution_result.evolved:
                scalar_values["evolution/elite_fitness"] = (
                    last_evolution_result.elite_fitness
                )
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
                if scheduler is not None:
                    scheduler.record_eval_return(last_eval_return)

            if scheduler is not None and step % scheduler_update_interval == 0:
                disagreement = compute_critic_disagreement(
                    actor=actor,
                    critic=critic,
                    replay_buffer=replay_buffer,
                    batch_size=min(256, batch_size),
                )
                previous_mode = current_mode
                last_scheduler_state = scheduler.update(step, disagreement)
                resources = scheduler.current_resources()
                current_mode = last_scheduler_state.mode
                current_mode_id = last_scheduler_state.mode_id
                current_update_ratio = resources.update_ratio
                current_pop_fraction = resources.pop_fraction
                current_active_heads = resources.active_ea_heads
                logger.scalar(
                    "scheduler/uncertainty_score",
                    last_scheduler_state.uncertainty_score,
                    step,
                )
                logger.scalar("scheduler/mode_id", current_mode_id, step)
                logger.scalar(
                    "scheduler/critic_disagreement",
                    last_scheduler_state.critic_disagreement,
                    step,
                )
                logger.scalar(
                    "scheduler/learning_progress",
                    last_scheduler_state.learning_progress,
                    step,
                )
                logger.scalar(
                    "scheduler/progress_need",
                    last_scheduler_state.progress_need,
                    step,
                )
                logger.scalar("scheduler/update_ratio", current_update_ratio, step)
                logger.scalar("scheduler/pop_fraction", current_pop_fraction, step)
                logger.scalar("scheduler/active_ea_heads", current_active_heads, step)
                trace_row = {
                    "step": step,
                    "mode": current_mode,
                    "mode_id": current_mode_id,
                    "uncertainty_score": last_scheduler_state.uncertainty_score,
                    "critic_disagreement": last_scheduler_state.critic_disagreement,
                    "learning_progress": last_scheduler_state.learning_progress,
                    "progress_need": last_scheduler_state.progress_need,
                    "update_ratio": current_update_ratio,
                    "pop_fraction": current_pop_fraction,
                    "active_ea_heads": current_active_heads,
                    "eval_return": last_eval_return,
                }
                scheduler_trace.append(trace_row)
                if current_mode != previous_mode:
                    mode_switches.append(
                        {
                            "step": step,
                            "from": previous_mode,
                            "to": current_mode,
                            "uncertainty_score": last_scheduler_state.uncertainty_score,
                        }
                    )

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
                    "B_pop": len(pop_buffer) if pop_buffer is not None else None,
                    "ea_return": last_ea_mean_return,
                    "ea_heads": last_ea_active_heads if last_ea_active_heads else None,
                    "filter_accept": last_filter_acceptance_rate,
                    "mode": current_mode if ba_enabled else None,
                    "elite": last_evolution_result.elite_head
                    if last_evolution_result and last_evolution_result.evolved
                    else None,
                    "updates": update_count,
                },
            )

        wall_clock_sec = time.perf_counter() - wall_clock_start
        scheduler_trace_path = dirs["logs"] / "scheduler_trace.jsonl"
        mode_switches_path = dirs["logs"] / "mode_switches.json"
        if scheduler is not None:
            _write_jsonl(scheduler_trace_path, scheduler_trace)
            _write_json(mode_switches_path, mode_switches)
        mode_fraction = {
            mode: count / total_steps for mode, count in mode_step_counts.items()
        }
        return {
            "run_name": run_name,
            "log_dir": str(dirs["logs"]),
            "model_dir": str(dirs["models"]),
            "latest_checkpoint": str(dirs["models"] / "latest.pt"),
            "root_latest_checkpoint": str(dirs["root"] / "models" / "latest.pt"),
            "total_steps": total_steps,
            "updates": update_count,
            "episodes": episode_idx,
            "wall_clock_sec": wall_clock_sec,
            "last_eval_return": last_eval_return,
            "final_B_rl_size": len(replay_buffer),
            "final_B_pop_size": len(pop_buffer) if pop_buffer is not None else 0,
            "last_ea_mean_return": last_ea_mean_return,
            "last_filter_acceptance_rate": last_filter_acceptance_rate,
            "current_mode": current_mode if ba_enabled else None,
            "mode_step_counts": mode_step_counts,
            "mode_fraction": mode_fraction,
            "mode_switches": mode_switches,
            "scheduler_trace_path": str(scheduler_trace_path)
            if scheduler is not None
            else None,
            "mode_switches_path": str(mode_switches_path)
            if scheduler is not None
            else None,
            "last_evolution_elite": last_evolution_result.elite_head
            if last_evolution_result and last_evolution_result.evolved
            else None,
            "last_evolution_mutated_heads": last_evolution_result.mutated_heads
            if last_evolution_result and last_evolution_result.evolved
            else [],
            "scheduler_uncertainty": last_scheduler_state.uncertainty_score
            if last_scheduler_state is not None
            else None,
            "device": str(device),
        }
    finally:
        logger.close()
        env.close()
