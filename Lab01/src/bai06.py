import gymnasium as gym
from collections import Counter

env = gym.make("CartPole-v1")
actions = []
for _ in range(20):
    action = env.action_space.sample()
    actions.append(action)
print("Actions:", actions)
action_frequency = Counter(actions)
print("Action frequency:", action_frequency)
env.close()