import gymnasium as gym
import highway_env
from stable_baselines3 import PPO


def main():
    env = gym.make("intersection-v0", render_mode="human")

    # env.configure({
    #     "simulation_frequency": 5,
    #     "policy_frequency": 1
    # })
    model = PPO.load("ppo_intersection_model")

    obs, info = env.reset()

    for step in range(500):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step: {step}, Action: {action}, Reward: {reward}")

        if terminated or truncated:
            print("Episode ended")
            obs, info = env.reset()

        env.close()

    # for step in range(300):
    #     action = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     time.sleep(0.05)  # 👈 slows down simulation

    #     if terminated or truncated:
    #         obs, info = env.reset()

    #     env.close()


if __name__ == "__main__":
    main()