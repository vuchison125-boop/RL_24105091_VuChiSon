import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
observation, info = env.reset(seed=42)
actions = [2, 2, 1, 1, 0, 1, 2, 2]
print(env.render())

for action in actions:
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Action: {action}")
    print(env.render())
    print(f"Reward: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Truncated: {truncated}")
    print()
    if terminated or truncated:
        break

env.close()