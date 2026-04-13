import argparse

import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


class NearMissPenaltyWrapper(gym.RewardWrapper):
    """Add a penalty when the ego vehicle gets too close to surrounding vehicles."""

    def __init__(self, env: gym.Env, unsafe_distance: float, penalty_scale: float):
        super().__init__(env)
        self.unsafe_distance = unsafe_distance
        self.penalty_scale = penalty_scale

    def reward(self, reward):
        ego = getattr(self.unwrapped, "vehicle", None)
        road = getattr(self.unwrapped, "road", None)
        if ego is None or road is None:
            return reward

        min_distance = float("inf")
        for vehicle in getattr(road, "vehicles", []):
            if vehicle is ego:
                continue
            distance = float(np.linalg.norm(ego.position - vehicle.position))
            min_distance = min(min_distance, distance)

        if min_distance < self.unsafe_distance:
            penalty = self.penalty_scale * (self.unsafe_distance - min_distance)
            return float(reward) - penalty
        return reward


class WaitingPenaltyWrapper(gym.RewardWrapper):
    """Add a small penalty when the ego vehicle is effectively waiting."""

    def __init__(self, env: gym.Env, wait_speed_threshold: float, wait_penalty: float):
        super().__init__(env)
        self.wait_speed_threshold = wait_speed_threshold
        self.wait_penalty = wait_penalty

    def reward(self, reward):
        ego = getattr(self.unwrapped, "vehicle", None)
        speed = float(getattr(ego, "speed", 0.0))
        if abs(speed) < self.wait_speed_threshold:
            return float(reward) - self.wait_penalty
        return reward


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on highway-env intersection-v0.")
    parser.add_argument("--env-id", default="intersection-v0", help="Gymnasium environment id.")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total PPO training timesteps.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path/name to save the trained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--reward-variant",
        default="baseline",
        choices=["baseline", "collision_strong", "near_miss_penalty", "waiting_penalty"],
        help="Reward variant to train.",
    )
    return parser.parse_args()


def apply_reward_variant(env: gym.Env, reward_variant: str) -> gym.Env:
    if reward_variant == "baseline":
        return env

    if reward_variant == "collision_strong":
        env.unwrapped.configure({"collision_reward": -10})
        return env

    if reward_variant == "near_miss_penalty":
        return NearMissPenaltyWrapper(
            env=env,
            unsafe_distance=10.0,
            penalty_scale=0.05,
        )

    if reward_variant == "waiting_penalty":
        return WaitingPenaltyWrapper(
            env=env,
            wait_speed_threshold=0.5,
            wait_penalty=0.02,
        )

    raise ValueError(f"Unknown reward variant: {reward_variant}")


def make_env(env_id: str, reward_variant: str):
    env = gym.make(env_id)
    env = apply_reward_variant(env, reward_variant)
    return env


def main():
    args = parse_args()
    model_path = args.model_path or f"ppo_intersection_model_{args.reward_variant}"
    env = make_env(args.env_id, args.reward_variant)

    try:
        # Optional: check whether the environment follows Gym API properly
        check_env(env, warn=True)

        env.reset(seed=args.seed)

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Seed:", args.seed)
        print("Timesteps:", args.timesteps)
        print("Reward variant:", args.reward_variant)

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=args.seed,
        )

        model.learn(total_timesteps=args.timesteps)

        model.save(model_path)
        print(f"Model saved as {model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
