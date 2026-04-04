import gymnasium as gym
import highway_env

env = gym.make("intersection-v0", render_mode="human")
obs, info = env.reset()

for step in range(300):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step: {step}, Action: {action}, Reward: {reward}")

    if terminated or truncated:
        print("Episode ended")
        obs, info = env.reset()

env.close()