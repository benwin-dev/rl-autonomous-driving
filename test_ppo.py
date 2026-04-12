import argparse

import gymnasium as gym
import highway_env
import pygame
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and visualize a trained PPO model.")
    parser.add_argument("--env-id", default="intersection-v0", help="Gymnasium environment id.")
    parser.add_argument(
        "--model-path",
        default="ppo_intersection_model",
        help="Path/name used by PPO.load().",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500,
        help="Total visualization steps (not episodes).",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic policy actions (default deterministic).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env = gym.make(args.env_id, render_mode="human")
    model = PPO.load(args.model_path)

    obs, info = env.reset()
    episode = 1
    episode_step = 0
    episode_reward = 0.0
    success_count = 0
    collision_count = 0
    timeout_count = 0

    hud_metrics = {
        "global_step": 0,
        "episode": episode,
        "episode_step": episode_step,
        "reward": 0.0,
        "episode_reward": episode_reward,
        "speed": 0.0,
        "success": success_count,
        "collision": collision_count,
        "timeout": timeout_count,
        "last_outcome": "N/A",
    }

    def draw_hud(agent_surface, _sim_surface):
        agent_surface.fill((25, 25, 25))
        font = pygame.font.Font(None, 30)
        lines = [
            "PPO Live Metrics",
            f"Global step: {hud_metrics['global_step']}",
            f"Episode: {hud_metrics['episode']}",
            f"Episode step: {hud_metrics['episode_step']}",
            f"Step reward: {hud_metrics['reward']:.3f}",
            f"Episode reward: {hud_metrics['episode_reward']:.3f}",
            f"Speed: {hud_metrics['speed']:.2f}",
            f"Success: {hud_metrics['success']}",
            f"Collision: {hud_metrics['collision']}",
            f"Timeout: {hud_metrics['timeout']}",
            f"Last outcome: {hud_metrics['last_outcome']}",
        ]
        y = 20
        for line in lines:
            text_surface = font.render(line, True, (235, 235, 235))
            agent_surface.blit(text_surface, (20, y))
            y += 32

    # Ensure viewer exists, then attach the HUD overlay.
    env.render()
    if env.unwrapped.viewer:
        env.unwrapped.viewer.set_agent_display(draw_hud)

    for step in range(args.steps):
        action, _states = model.predict(obs, deterministic=not args.stochastic)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_step += 1
        episode_reward += reward
        speed = float(info.get("speed", 0.0))

        hud_metrics.update(
            {
                "global_step": step,
                "episode": episode,
                "episode_step": episode_step,
                "reward": float(reward),
                "episode_reward": float(episode_reward),
                "speed": speed,
            }
        )

        print(
            f"GlobalStep: {step} | Episode: {episode} | EpisodeStep: {episode_step} "
            f"| Action: {action} | Reward: {reward:.3f} | EpisodeReward: {episode_reward:.3f} "
            f"| Speed: {speed:.2f}"
        )

        if terminated or truncated:
            crashed = bool(info.get("crashed", False))
            if crashed:
                outcome = "COLLISION"
                collision_count += 1
            elif truncated:
                outcome = "TIMEOUT"
                timeout_count += 1
            else:
                outcome = "SUCCESS"
                success_count += 1

            hud_metrics.update(
                {
                    "success": success_count,
                    "collision": collision_count,
                    "timeout": timeout_count,
                    "last_outcome": outcome,
                }
            )

            print(
                f"Episode {episode} ended: {outcome} | "
                f"EpisodeSteps: {episode_step} | TotalEpisodeReward: {episode_reward:.3f} | "
                f"RunningTotals -> Success: {success_count}, Collision: {collision_count}, Timeout: {timeout_count}"
            )
            obs, info = env.reset()
            episode += 1
            episode_step = 0
            episode_reward = 0.0

    completed_episodes = success_count + collision_count + timeout_count
    if completed_episodes > 0:
        print("\n=== Final Summary ===")
        print(f"Completed episodes: {completed_episodes}")
        print(f"Success: {success_count} ({100.0 * success_count / completed_episodes:.2f}%)")
        print(f"Collision: {collision_count} ({100.0 * collision_count / completed_episodes:.2f}%)")
        print(f"Timeout: {timeout_count} ({100.0 * timeout_count / completed_episodes:.2f}%)")

    env.close()


if __name__ == "__main__":
    main()
