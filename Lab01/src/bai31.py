import gymnasium as gym
import numpy as np

def angle_based_policy(observation):
    pole_angle = observation[2]
    if pole_angle < 0:
        return 0
    else:
        return 1

def evaluate_policy(env, policy, n_episodes=100):
    rewards = []
    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0
        for _ in range(500):
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

env = gym.make("CartPole-v1")
mean_reward = evaluate_policy(env, angle_based_policy)
print(f"Angle-based mean reward: {mean_reward:.2f}")
env.close()