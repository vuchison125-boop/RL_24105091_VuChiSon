import gymnasium as gym
import numpy as np

def random_agent(env, max_steps=500):
    observation, info = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


env = gym.make("CartPole-v1")
episode_rewards = []
for _ in range(100):
    reward = random_agent(env)
    episode_rewards.append(reward)

mean_reward = np.mean(episode_rewards)
min_reward = np.min(episode_rewards)
max_reward = np.max(episode_rewards)
std_reward = np.std(episode_rewards)

print(f"Mean reward : {mean_reward:.2f}")
print(f"Min reward  : {min_reward:.2f}")
print(f"Max reward  : {max_reward:.2f}")
print(f"Std reward  : {std_reward:.2f}")

env.close()