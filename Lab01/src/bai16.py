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
    return total_reward, step + 1

env = gym.make("CartPole-v1")
episode_rewards = []
episode_lengths = []

for _ in range(100):
    reward, length = random_agent(env)
    episode_rewards.append(reward)
    episode_lengths.append(length)
best_index = np.argmax(episode_rewards)

print("Best episode:", best_index + 1)
print("Best reward:", episode_rewards[best_index])
print("Best episode length:", episode_lengths[best_index])

env.close()