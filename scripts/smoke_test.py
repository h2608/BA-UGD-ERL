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
    parser.add_argument("--stage", choices=["env", "td3"], required=True)
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--total_steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.stage == "env":
        result = run_env_smoke(config)
    else:
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

    print(f"{args.stage} smoke test finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
