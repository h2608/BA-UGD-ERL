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


SCAN_CANDIDATES: list[dict[str, Any]] = [
    {
        "name": "default_like",
        "explore_enter": 0.65,
        "explore_exit": 0.55,
        "exploit_enter": 0.35,
        "exploit_exit": 0.45,
        "min_mode_steps": 5000,
        "progress_scale": 10.0,
        "disagreement_weight": 0.5,
        "progress_weight": 0.5,
    },
    {
        "name": "easy_explore",
        "explore_enter": 0.50,
        "explore_exit": 0.45,
        "exploit_enter": 0.25,
        "exploit_exit": 0.35,
        "min_mode_steps": 2500,
        "progress_scale": 10.0,
        "disagreement_weight": 0.6,
        "progress_weight": 0.4,
    },
    {
        "name": "easy_exploit",
        "explore_enter": 0.80,
        "explore_exit": 0.70,
        "exploit_enter": 0.60,
        "exploit_exit": 0.70,
        "min_mode_steps": 2500,
        "progress_scale": 50.0,
        "disagreement_weight": 0.2,
        "progress_weight": 0.8,
    },
    {
        "name": "fast_switch_balanced",
        "explore_enter": 0.55,
        "explore_exit": 0.48,
        "exploit_enter": 0.45,
        "exploit_exit": 0.52,
        "min_mode_steps": 1000,
        "progress_scale": 25.0,
        "disagreement_weight": 0.5,
        "progress_weight": 0.5,
    },
    {
        "name": "progress_dominant",
        "explore_enter": 0.58,
        "explore_exit": 0.50,
        "exploit_enter": 0.42,
        "exploit_exit": 0.50,
        "min_mode_steps": 2500,
        "progress_scale": 100.0,
        "disagreement_weight": 0.25,
        "progress_weight": 0.75,
    },
    {
        "name": "disagreement_dominant",
        "explore_enter": 0.55,
        "explore_exit": 0.48,
        "exploit_enter": 0.35,
        "exploit_exit": 0.45,
        "min_mode_steps": 2500,
        "progress_scale": 10.0,
        "disagreement_weight": 0.75,
        "progress_weight": 0.25,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan Hopper scheduler parameters.")
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--total_steps", type=int, default=20000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--eval_episodes", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=5000)
    parser.add_argument("--checkpoint_interval", type=int, default=20000)
    parser.add_argument("--console_interval", type=int, default=5000)
    parser.add_argument("--max_rollout_steps", type=int, default=1000)
    parser.add_argument("--filter_margin", type=float, default=20.0)
    parser.add_argument("--candidates", nargs="+", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Defaults to outputs/results/scheduler_scan_<timestamp>.jsonl",
    )
    return parser.parse_args()


def build_overrides(
    args: argparse.Namespace, candidate: dict[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "experiment": {
            "algorithm": "ba_ugd_erl",
            "name": f"hopper_schedscan_{candidate['name']}",
            "seed": seed,
            "total_steps": args.total_steps,
        },
        "env": {"eval_episodes": args.eval_episodes},
        "logging": {
            "eval_interval": args.eval_interval,
            "checkpoint_interval": args.checkpoint_interval,
            "console_interval": args.console_interval,
        },
        "scheduler": {
            "enabled": True,
            "strategy": "uncertainty",
            "update_interval": 5000,
            "initial_mode": "Hybrid",
            **{key: value for key, value in candidate.items() if key != "name"},
        },
        "ba_ugd_erl": {
            "enabled": True,
            "num_ea_heads": 4,
            "mixed_sampling": {"enabled": True, "pop_fraction": 0.5},
            "filter": {
                "enabled": True,
                "warmup_episodes": 5,
                "return_margin": args.filter_margin,
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
                "max_rollout_steps": args.max_rollout_steps,
            },
        },
    }


def selected_candidates(names: list[str] | None) -> list[dict[str, Any]]:
    if names is None:
        return SCAN_CANDIDATES
    by_name = {candidate["name"]: candidate for candidate in SCAN_CANDIDATES}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"Unknown candidate names: {missing}")
    return [by_name[name] for name in names]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    candidates = selected_candidates(args.candidates)
    planned = [(candidate, seed) for candidate in candidates for seed in args.seeds]
    output_path = Path(
        args.output_jsonl
        or f"outputs/results/scheduler_scan_{int(time.time())}.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Planned scheduler scan runs: {len(planned)}")
    for candidate, seed in planned:
        print(
            f"  candidate={candidate['name']} seed={seed} "
            f"steps={args.total_steps}"
        )
    if args.dry_run:
        return

    with open(output_path, "a", encoding="utf-8") as f:
        for candidate, seed in planned:
            result = run_training(config, overrides=build_overrides(args, candidate, seed))
            row = {
                "scan_candidate": candidate["name"],
                "seed": seed,
                "total_steps": args.total_steps,
                "scheduler_params": candidate,
                **result,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            switches = result.get("mode_switches", [])
            print(
                f"finished candidate={candidate['name']} seed={seed} "
                f"switches={len(switches)} mode_fraction={result.get('mode_fraction')}"
            )
    print(f"Scan results written to {output_path}")


if __name__ == "__main__":
    main()
