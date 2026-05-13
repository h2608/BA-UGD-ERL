from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training import run_training
from core.utils import load_config


VARIANTS = [
    "td3_only",
    "ba_static_switch",
    "ba_static_switch_no_filter",
    "ba_scheduler",
    "ba_scheduler_no_filter",
    "ba_scheduler_easy_exploit",
    "ba_scheduler_easy_exploit_no_filter",
    "ba_scheduler_disagreement_dominant",
    "ba_scheduler_disagreement_dominant_no_filter",
]


SCHEDULER_CANDIDATES: dict[str, dict[str, float | int]] = {
    "default": {
        "explore_enter": 0.65,
        "explore_exit": 0.55,
        "exploit_enter": 0.35,
        "exploit_exit": 0.45,
        "min_mode_steps": 5000,
        "progress_scale": 10.0,
        "disagreement_weight": 0.5,
        "progress_weight": 0.5,
    },
    "easy_exploit": {
        "explore_enter": 0.80,
        "explore_exit": 0.70,
        "exploit_enter": 0.60,
        "exploit_exit": 0.70,
        "min_mode_steps": 2500,
        "progress_scale": 50.0,
        "disagreement_weight": 0.2,
        "progress_weight": 0.8,
    },
    "disagreement_dominant": {
        "explore_enter": 0.55,
        "explore_exit": 0.48,
        "exploit_enter": 0.35,
        "exploit_exit": 0.45,
        "min_mode_steps": 2500,
        "progress_scale": 10.0,
        "disagreement_weight": 0.75,
        "progress_weight": 0.25,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hopper comparison experiments.")
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--total_steps", type=int, default=100000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--checkpoint_interval", type=int, default=50000)
    parser.add_argument("--console_interval", type=int, default=5000)
    parser.add_argument("--max_rollout_steps", type=int, default=1000)
    parser.add_argument("--filter_margin", type=float, default=20.0)
    parser.add_argument("--max_runs", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Defaults to outputs/results/hopper_comparison_<timestamp>.jsonl",
    )
    return parser.parse_args()


def ba_overrides(
    *,
    variant: str,
    seed: int,
    total_steps: int,
    eval_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    console_interval: int,
    max_rollout_steps: int,
    filter_margin: float,
) -> dict[str, Any]:
    filter_enabled = not variant.endswith("_no_filter")
    scheduler_strategy = "static_switch" if "static_switch" in variant else "uncertainty"
    scheduler_params = SCHEDULER_CANDIDATES["default"]
    if "easy_exploit" in variant:
        scheduler_params = SCHEDULER_CANDIDATES["easy_exploit"]
    elif "disagreement_dominant" in variant:
        scheduler_params = SCHEDULER_CANDIDATES["disagreement_dominant"]
    return {
        "experiment": {
            "algorithm": "ba_ugd_erl",
            "name": f"hopper_{variant}",
            "seed": seed,
            "total_steps": total_steps,
        },
        "env": {"eval_episodes": eval_episodes},
        "logging": {
            "eval_interval": eval_interval,
            "checkpoint_interval": checkpoint_interval,
            "console_interval": console_interval,
        },
        "scheduler": {
            "enabled": True,
            "strategy": scheduler_strategy,
            "update_interval": 5000,
            "explore_fraction": 0.25,
            "initial_mode": "Hybrid",
            **scheduler_params,
        },
        "ba_ugd_erl": {
            "enabled": True,
            "num_ea_heads": 4,
            "mixed_sampling": {"enabled": True, "pop_fraction": 0.5},
            "filter": {
                "enabled": filter_enabled,
                "warmup_episodes": 5,
                "return_margin": filter_margin,
            },
            "evolution": {
                "enabled": True,
                "interval": 5000,
                "mutation_std": 0.05,
            },
            "ea": {
                "rollout_enabled": True,
                "rollout_interval": 5000,
                "rollout_episodes_per_head": 1,
                "active_heads": 4,
                "max_rollout_steps": max_rollout_steps,
            },
        },
    }


def td3_overrides(
    *,
    seed: int,
    total_steps: int,
    eval_episodes: int,
    eval_interval: int,
    checkpoint_interval: int,
    console_interval: int,
) -> dict[str, Any]:
    return {
        "experiment": {
            "algorithm": "td3_only",
            "name": "hopper_td3_only",
            "seed": seed,
            "total_steps": total_steps,
        },
        "env": {"eval_episodes": eval_episodes},
        "logging": {
            "eval_interval": eval_interval,
            "checkpoint_interval": checkpoint_interval,
            "console_interval": console_interval,
        },
        "scheduler": {"enabled": False},
        "ba_ugd_erl": {"enabled": False},
    }


def build_overrides(args: argparse.Namespace, variant: str, seed: int) -> dict[str, Any]:
    if variant == "td3_only":
        return td3_overrides(
            seed=seed,
            total_steps=args.total_steps,
            eval_episodes=args.eval_episodes,
            eval_interval=args.eval_interval,
            checkpoint_interval=args.checkpoint_interval,
            console_interval=args.console_interval,
        )
    return ba_overrides(
        variant=variant,
        seed=seed,
        total_steps=args.total_steps,
        eval_episodes=args.eval_episodes,
        eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        console_interval=args.console_interval,
        max_rollout_steps=args.max_rollout_steps,
        filter_margin=args.filter_margin,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_path = Path(
        args.output_jsonl
        or f"outputs/results/hopper_comparison_{int(time.time())}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    planned = [
        (variant, seed)
        for variant in args.variants
        for seed in args.seeds
    ]
    if args.max_runs is not None:
        planned = planned[: args.max_runs]

    print(f"Planned runs: {len(planned)}")
    for variant, seed in planned:
        print(f"  variant={variant} seed={seed} total_steps={args.total_steps}")

    if args.dry_run:
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for variant, seed in planned:
            overrides = build_overrides(args, variant, seed)
            result = run_training(config, overrides=overrides)
            row = {
                "variant": variant,
                "seed": seed,
                "total_steps": args.total_steps,
                **result,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"finished variant={variant} seed={seed} "
                f"eval={result.get('last_eval_return')}"
            )
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()
