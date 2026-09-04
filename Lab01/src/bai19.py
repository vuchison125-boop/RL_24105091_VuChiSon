import gymnasium as gym
import numpy as np

observations = []

for _ in range(10):
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    observations.append(observation)
    env.close()

for i, observation in enumerate(observations, start=1):
    print(f"Environment {i}: {observation}")

all_same = all(
    np.array_equal(observations[0], observation)
    for observation in observations[1:]
)

print("All observations are identical:", all_same)

# Khi sử dụng cùng seed, các environment độc lập tạo ra cùng initial observation.
# Điều này cho phép tái lập kết quả thí nghiệm.