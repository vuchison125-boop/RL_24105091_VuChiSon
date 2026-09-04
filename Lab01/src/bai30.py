import gymnasium as gym
import numpy as np

def always_left_policy(observation):
    return 0

def always_right_policy(observation):
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

left_mean = evaluate_policy(env, always_left_policy)
right_mean = evaluate_policy(env, always_right_policy)

print(f"Always left mean reward: {left_mean:.2f}")
print(f"Always right mean reward: {right_mean:.2f}")

env.close()