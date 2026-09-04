import gymnasium as gym
import numpy as np

def experiment(seed, n_episodes):
    env = gym.make("CartPole-v1")
    env.action_space.seed(seed)
    rewards = []
    for _ in range(n_episodes):
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(500):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    result = {
        "seed": seed,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "max_reward": np.max(rewards),
        "min_reward": np.min(rewards)
    }
    env.close()
    return result

seeds = [42, 100, 200, 300, 400]

for seed in seeds:
    result = experiment(seed, 20)
    print(
        f"Seed: {result['seed']}, "
        f"Mean: {result['mean_reward']:.2f}, "
        f"Std: {result['std_reward']:.2f}, "
        f"Max: {result['max_reward']:.2f}, "
        f"Min: {result['min_reward']:.2f}"
    )