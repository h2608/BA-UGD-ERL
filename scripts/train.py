from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training import run_training
from core.utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TD3-only baseline.")
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    overrides = {}
    if args.total_steps is not None:
        overrides.setdefault("experiment", {})["total_steps"] = args.total_steps
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed
    result = run_training(config, overrides=overrides)
    print("Training finished:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
