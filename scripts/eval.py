from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training import build_td3_components, evaluate_actor
from core.utils import get_device, load_config, make_env, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a TD3 actor checkpoint.")
    parser.add_argument("--config", type=str, default="configs/hopper.yaml")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/latest.pt")
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "config" in checkpoint:
        config = checkpoint["config"]
    seed = int(config["experiment"].get("seed", 0))
    set_seed(seed)
    device = get_device()

    env = make_env(config["env"]["name"], seed=seed)
    try:
        components = build_td3_components(config, env, device)
        actor = components["actor"]
        actor.load_state_dict(checkpoint["actor"])
    finally:
        env.close()

    episodes = args.episodes or int(config["env"].get("eval_episodes", 5))
    avg_return = evaluate_actor(
        actor=actor,
        env_name=config["env"]["name"],
        seed=seed + 50000,
        episodes=episodes,
        device=device,
    )
    print(f"Average return over {episodes} episodes: {avg_return:.3f}")


if __name__ == "__main__":
    main()
