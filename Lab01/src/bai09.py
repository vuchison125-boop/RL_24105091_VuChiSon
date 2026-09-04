import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for t in range(20):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print("t:", t, "action:", action, "reward:", reward)
    if terminated or truncated:
        break

env.close()