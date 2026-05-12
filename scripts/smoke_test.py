from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training import run_env_smoke, run_training
from core.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run project smoke tests.")
    parser.add_argument(
        "--stage",
        choices=[
            "env",
            "td3",
            "ba_rollout",
            "ba_mixed",
            "ba_filter",
            "ba_scheduler",
        ],
        required=True,
    )
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--total_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "env":
        result = run_env_smoke(config)
    elif args.stage == "td3":
        total_steps = args.total_steps if args.total_steps is not None else 6000
        result = run_training(
            config,
            overrides={
                "experiment": {"total_steps": total_steps},
                "env": {"eval_episodes": 1},
                "logging": {
                    "eval_interval": 1000,
                    "checkpoint_interval": max(1000, total_steps),
                    "console_interval": 1000,
                },
            },
        )
    elif args.stage == "ba_rollout":
        total_steps = args.total_steps if args.total_steps is not None else 6000
        result = run_training(
            config,
            overrides={
                "experiment": {
                    "algorithm": "ba_ugd_erl",
                    "name": "hopper_ba_rollout_smoke",
                    "total_steps": total_steps,
                },
                "env": {"eval_episodes": 1},
                "logging": {
                    "eval_interval": 1000,
                    "checkpoint_interval": max(1000, total_steps),
                    "console_interval": 1000,
                },
                "ba_ugd_erl": {
                    "enabled": True,
                    "num_ea_heads": 4,
                    "mixed_sampling": {"enabled": False},
                    "filter": {"enabled": False},
                    "ea": {
                        "rollout_enabled": True,
                        "rollout_interval": 1000,
                        "rollout_episodes_per_head": 1,
                        "active_heads": 4,
                        "max_rollout_steps": 100,
                    },
                },
            },
        )
    elif args.stage == "ba_mixed":
        total_steps = args.total_steps if args.total_steps is not None else 6000
        result = run_training(
            config,
            overrides={
                "experiment": {
                    "algorithm": "ba_ugd_erl",
                    "name": "hopper_ba_mixed_smoke",
                    "total_steps": total_steps,
                },
                "env": {"eval_episodes": 1},
                "logging": {
                    "eval_interval": 1000,
                    "checkpoint_interval": max(1000, total_steps),
                    "console_interval": 1000,
                },
                "ba_ugd_erl": {
                    "enabled": True,
                    "num_ea_heads": 4,
                    "mixed_sampling": {"enabled": True, "pop_fraction": 0.5},
                    "filter": {"enabled": False},
                    "ea": {
                        "rollout_enabled": True,
                        "rollout_interval": 1000,
                        "rollout_episodes_per_head": 1,
                        "active_heads": 4,
                        "max_rollout_steps": 100,
                    },
                },
            },
        )
    elif args.stage == "ba_filter":
        total_steps = args.total_steps if args.total_steps is not None else 6000
        result = run_training(
            config,
            overrides={
                "experiment": {
                    "algorithm": "ba_ugd_erl",
                    "name": "hopper_ba_filter_smoke",
                    "total_steps": total_steps,
                },
                "env": {"eval_episodes": 1},
                "logging": {
                    "eval_interval": 1000,
                    "checkpoint_interval": max(1000, total_steps),
                    "console_interval": 1000,
                },
                "ba_ugd_erl": {
                    "enabled": True,
                    "num_ea_heads": 4,
                    "mixed_sampling": {"enabled": True, "pop_fraction": 0.5},
                    "filter": {
                        "enabled": True,
                        "warmup_episodes": 5,
                        "return_margin": 100.0,
                    },
                    "ea": {
                        "rollout_enabled": True,
                        "rollout_interval": 1000,
                        "rollout_episodes_per_head": 1,
                        "active_heads": 4,
                        "max_rollout_steps": 100,
                    },
                },
            },
        )
    else:
        total_steps = args.total_steps if args.total_steps is not None else 6000
        result = run_training(
            config,
            overrides={
                "experiment": {
                    "algorithm": "ba_ugd_erl",
                    "name": "hopper_ba_scheduler_smoke",
                    "total_steps": total_steps,
                },
                "env": {"eval_episodes": 1},
                "logging": {
                    "eval_interval": 1000,
                    "checkpoint_interval": max(1000, total_steps),
                    "console_interval": 1000,
                },
                "scheduler": {"enabled": True, "update_interval": 5000},
                "ba_ugd_erl": {
                    "enabled": True,
                    "num_ea_heads": 4,
                    "mixed_sampling": {"enabled": True, "pop_fraction": 0.5},
                    "filter": {
                        "enabled": True,
                        "warmup_episodes": 5,
                        "return_margin": 100.0,
                    },
                    "ea": {
                        "rollout_enabled": True,
                        "rollout_interval": 1000,
                        "rollout_episodes_per_head": 1,
                        "active_heads": 4,
                        "max_rollout_steps": 100,
                    },
                },
            },
        )

    print(f"{args.stage} smoke test finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
