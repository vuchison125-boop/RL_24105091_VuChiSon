import gymnasium as gym

env = gym.make("CliffWalking-v1")

n_states = env.observation_space.n
n_actions = env.action_space.n

print("States:", n_states)
print("Actions:", n_actions)

env.close()