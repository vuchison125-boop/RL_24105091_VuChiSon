import gymnasium as gym

ACTION_NAMES = {
    0: "LEFT",
    1: "DOWN",
    2: "RIGHT",
    3: "UP"
}

env = gym.make("FrozenLake-v1", is_slippery=False)
observation, info = env.reset(seed=42)
action = env.action_space.sample()

print("Action:", action)
print(f"Action {action} -> {ACTION_NAMES[action]}")

env.close()