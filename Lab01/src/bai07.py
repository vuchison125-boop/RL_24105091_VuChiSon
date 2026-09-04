import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

print("State before action:", observation)
action = env.action_space.sample()
next_observation, reward, terminated, truncated, info = env.step(action)

print("Action:", action)
print("State after action:", next_observation)
print("Reward:", reward)
print("Terminated:", terminated)
print("Truncated:", truncated)
print("Info:", info)

env.close()