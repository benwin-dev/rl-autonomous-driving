import argparse

import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


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
        default="ppo_intersection_model",
        help="Path/name to save the trained model.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def make_env(env_id: str):
    env = gym.make(env_id)
    return env


def main():
    args = parse_args()
    env = make_env(args.env_id)

    try:
        # Optional: check whether the environment follows Gym API properly
        check_env(env, warn=True)

        env.reset(seed=args.seed)

        print("Observation space:", env.observation_space)
        print("Action space:", env.action_space)
        print("Seed:", args.seed)
        print("Timesteps:", args.timesteps)

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

        model.save(args.model_path)
        print(f"Model saved as {args.model_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
