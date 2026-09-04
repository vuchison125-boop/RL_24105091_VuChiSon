import gymnasium as gym
import numpy as np

def random_policy(observation, env):
    return env.action_space.sample()

def evaluate_policy(
    env_name,
    policy,
    n_episodes=100,
    seed=42
):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    for episode in range(n_episodes):
        observation, info = env.reset(seed=seed + episode)
        total_reward = 0
        for step in range(1000):
            action = policy(observation, env)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
        lengths.append(step + 1)
    result = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
        "mean_length": np.mean(lengths)
    }
    env.close()
    return result

result = evaluate_policy(
    "CartPole-v1",
    random_policy,
    n_episodes=100,
    seed=42
)

print(result)