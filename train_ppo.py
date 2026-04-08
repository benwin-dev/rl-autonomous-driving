import gymnasium as gym
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env


def make_env():
    env = gym.make("intersection-v0")
    return env


def main():
    env = make_env()

    # Optional: check whether the environment follows Gym API properly
    check_env(env, warn=True)

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

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
    )

    model.learn(total_timesteps=50000)

    model.save("ppo_intersection_model")
    print("Model saved as ppo_intersection_model")

    env.close()


if __name__ == "__main__":
    main()