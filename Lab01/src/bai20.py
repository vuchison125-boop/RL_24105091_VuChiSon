import gymnasium as gym
import numpy as np

def random_agent(env, seed, max_steps=500):
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    for _ in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward


env = gym.make("CartPole-v1")
rewards_42 = []
rewards_100 = []

for _ in range(20):
    rewards_42.append(random_agent(env, 42))

for _ in range(20):
    rewards_100.append(random_agent(env, 100))
mean_42 = np.mean(rewards_42)
mean_100 = np.mean(rewards_100)

print(f"Seed 42 mean reward: {mean_42:.2f}")
print(f"Seed 100 mean reward: {mean_100:.2f}")

env.close()