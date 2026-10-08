import gymnasium as gym
import numpy as np

env = gym.make("Taxi-v4")

n_states = env.observation_space.n
n_actions = env.action_space.n

policy = np.random.randint(n_actions, size=n_states)

print("Number of States:", n_states)
print("Number of Actions:", n_actions)
print("Random Policy:")
print(policy)

env.close()