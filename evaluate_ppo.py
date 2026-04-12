import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO


@dataclass
class EvalSummary:
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_reward: float
    avg_steps: float
    avg_waiting_steps: float
    avg_waiting_seconds: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO on highway-env intersection-v0 and compare with a random baseline."
    )
    parser.add_argument("--env-id", default="intersection-v0", help="Gymnasium environment id.")
    parser.add_argument(
        "--model-path",
        default="ppo_intersection_model",
        help="Path/name used by stable-baselines3 PPO.load().",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes per policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. Episode i uses seed+ i.",
    )
    parser.add_argument(
        "--wait-speed-threshold",
        type=float,
        default=0.5,
        help="Ego speed threshold below which a step counts as waiting.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap on steps per episode.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic PPO actions (default is deterministic).",
    )
    parser.add_argument(
        "--no-random-baseline",
        action="store_true",
        help="Disable random baseline evaluation.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save evaluation results as CSV.",
    )
    return parser.parse_args()


def episode_outcome(env: gym.Env, info: dict, terminated: bool, truncated: bool) -> tuple[bool, bool, bool]:
    crashed = bool(info.get("crashed", False))

    arrived = False
    unwrapped = env.unwrapped
    has_arrived_fn = getattr(unwrapped, "has_arrived", None)
    controlled_vehicles = getattr(unwrapped, "controlled_vehicles", [])
    if callable(has_arrived_fn) and controlled_vehicles:
        arrived = all(has_arrived_fn(vehicle) for vehicle in controlled_vehicles)

    success = (terminated or truncated) and (not crashed) and arrived
    collision = crashed
    timeout = truncated and not success and not collision
    return success, collision, timeout


def evaluate_policy(
    env_id: str,
    episodes: int,
    seed: int,
    wait_speed_threshold: float,
    action_selector,
    max_steps: int | None = None,
) -> EvalSummary:
    env = gym.make(env_id)
    policy_frequency = float(env.unwrapped.config.get("policy_frequency", 1.0))

    successes = 0
    collisions = 0
    timeouts = 0
    total_rewards = []
    total_steps = []
    waiting_steps_all = []

    for episode_idx in range(episodes):
        obs, info = env.reset(seed=seed + episode_idx)
        done = False
        forced_stop = False

        episode_reward = 0.0
        episode_steps = 0
        waiting_steps = 0

        while not done:
            action = action_selector(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += float(reward)
            episode_steps += 1

            speed = float(info.get("speed", 0.0))
            if abs(speed) < wait_speed_threshold:
                waiting_steps += 1

            if max_steps is not None and episode_steps >= max_steps and not done:
                forced_stop = True
                break

        if forced_stop:
            success = False
            collision = bool(info.get("crashed", False))
            timeout = not collision
        else:
            success, collision, timeout = episode_outcome(env, info, terminated, truncated)

        successes += int(success)
        collisions += int(collision)
        timeouts += int(timeout)

        total_rewards.append(episode_reward)
        total_steps.append(episode_steps)
        waiting_steps_all.append(waiting_steps)

    env.close()

    total_rewards_np = np.asarray(total_rewards, dtype=np.float64)
    total_steps_np = np.asarray(total_steps, dtype=np.float64)
    waiting_steps_np = np.asarray(waiting_steps_all, dtype=np.float64)

    return EvalSummary(
        episodes=episodes,
        success_rate=100.0 * successes / episodes,
        collision_rate=100.0 * collisions / episodes,
        timeout_rate=100.0 * timeouts / episodes,
        avg_reward=float(total_rewards_np.mean()),
        avg_steps=float(total_steps_np.mean()),
        avg_waiting_steps=float(waiting_steps_np.mean()),
        avg_waiting_seconds=float((waiting_steps_np / policy_frequency).mean()),
    )


def print_summary(title: str, summary: EvalSummary) -> None:
    print(f"\n=== {title} ===")
    print(f"Episodes: {summary.episodes}")
    print(f"Success rate: {summary.success_rate:.2f}%")
    print(f"Collision rate: {summary.collision_rate:.2f}%")
    print(f"Timeout rate: {summary.timeout_rate:.2f}%")
    print(f"Average reward: {summary.avg_reward:.3f}")
    print(f"Average episode length (steps): {summary.avg_steps:.2f}")
    print(f"Average waiting (steps): {summary.avg_waiting_steps:.2f}")
    print(f"Average waiting (seconds): {summary.avg_waiting_seconds:.2f}")


def print_comparison(ppo_summary: EvalSummary, random_summary: EvalSummary) -> None:
    print("\n=== PPO vs Random Delta (PPO - Random) ===")
    print(f"Success rate delta: {ppo_summary.success_rate - random_summary.success_rate:+.2f} pp")
    print(f"Collision rate delta: {ppo_summary.collision_rate - random_summary.collision_rate:+.2f} pp")
    print(f"Timeout rate delta: {ppo_summary.timeout_rate - random_summary.timeout_rate:+.2f} pp")
    print(f"Average reward delta: {ppo_summary.avg_reward - random_summary.avg_reward:+.3f}")
    print(f"Average waiting seconds delta: {ppo_summary.avg_waiting_seconds - random_summary.avg_waiting_seconds:+.2f}")


def _summary_to_dict(summary: EvalSummary) -> dict:
    return {
        "episodes": summary.episodes,
        "success_rate_percent": summary.success_rate,
        "collision_rate_percent": summary.collision_rate,
        "timeout_rate_percent": summary.timeout_rate,
        "avg_reward": summary.avg_reward,
        "avg_steps": summary.avg_steps,
        "avg_waiting_steps": summary.avg_waiting_steps,
        "avg_waiting_seconds": summary.avg_waiting_seconds,
    }


def save_json_report(
    output_path: str,
    args: argparse.Namespace,
    ppo_summary: EvalSummary,
    random_summary: EvalSummary | None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "environment": args.env_id,
        "model_path": args.model_path,
        "episodes_per_policy": args.episodes,
        "seed": args.seed,
        "wait_speed_threshold": args.wait_speed_threshold,
        "max_steps": args.max_steps,
        "ppo_deterministic": not args.stochastic,
        "metrics": {
            "ppo": _summary_to_dict(ppo_summary),
            "random_baseline": _summary_to_dict(random_summary) if random_summary is not None else None,
        },
    }

    if random_summary is not None:
        payload["deltas_ppo_minus_random"] = {
            "success_rate_delta_pp": ppo_summary.success_rate - random_summary.success_rate,
            "collision_rate_delta_pp": ppo_summary.collision_rate - random_summary.collision_rate,
            "timeout_rate_delta_pp": ppo_summary.timeout_rate - random_summary.timeout_rate,
            "avg_reward_delta": ppo_summary.avg_reward - random_summary.avg_reward,
            "avg_waiting_seconds_delta": ppo_summary.avg_waiting_seconds - random_summary.avg_waiting_seconds,
        }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved JSON report: {path}")


def save_csv_report(output_path: str, ppo_summary: EvalSummary, random_summary: EvalSummary | None) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "policy",
        "episodes",
        "success_rate_percent",
        "collision_rate_percent",
        "timeout_rate_percent",
        "avg_reward",
        "avg_steps",
        "avg_waiting_steps",
        "avg_waiting_seconds",
    ]

    rows = []
    rows.append({"policy": "ppo", **_summary_to_dict(ppo_summary)})
    if random_summary is not None:
        rows.append({"policy": "random_baseline", **_summary_to_dict(random_summary)})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV report: {path}")


def main() -> None:
    args = parse_args()

    model = PPO.load(args.model_path)
    deterministic = not args.stochastic

    def ppo_action_selector(obs, _env):
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    ppo_summary = evaluate_policy(
        env_id=args.env_id,
        episodes=args.episodes,
        seed=args.seed,
        wait_speed_threshold=args.wait_speed_threshold,
        action_selector=ppo_action_selector,
        max_steps=args.max_steps,
    )
    print_summary("PPO Policy", ppo_summary)

    random_summary = None
    if not args.no_random_baseline:

        def random_action_selector(_obs, env):
            return env.action_space.sample()

        random_summary = evaluate_policy(
            env_id=args.env_id,
            episodes=args.episodes,
            seed=args.seed,
            wait_speed_threshold=args.wait_speed_threshold,
            action_selector=random_action_selector,
            max_steps=args.max_steps,
        )
        print_summary("Random Baseline", random_summary)
        print_comparison(ppo_summary, random_summary)

    if args.output_json:
        save_json_report(args.output_json, args, ppo_summary, random_summary)
    if args.output_csv:
        save_csv_report(args.output_csv, ppo_summary, random_summary)


if __name__ == "__main__":
    main()
