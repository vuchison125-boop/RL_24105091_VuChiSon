import gymnasium as gym


env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

total_reward = 0.0

for t in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print("Episode length:", t + 1)
print("Total reward:", total_reward)

env.close()