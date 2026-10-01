import gymnasium as gym

env = gym.make("CartPole-v1")
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
print("Dtype:", env.observation_space.dtype)
print("Lower bound:", env.observation_space.low)
print("Upper bound:", env.observation_space.high)
env.close()