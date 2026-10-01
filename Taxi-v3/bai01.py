import gymnasium as gym

env = gym.make("Taxi-v4")

print("States:", env.observation_space.n)
print("Actions:", env.action_space.n)

env.close()