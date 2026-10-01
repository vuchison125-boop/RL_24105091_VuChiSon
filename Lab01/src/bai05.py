import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
print("Observation:", observation)
print("Type:", type(observation))
print("Shape:", observation.shape)
print("Info:", info)
env.close()
# Mỗi phần tử của observation là số thực kiểu numpy.float32