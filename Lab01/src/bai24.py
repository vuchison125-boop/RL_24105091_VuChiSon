import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")
observation, info = env.reset(seed=42)
print(env.render())

env.close()