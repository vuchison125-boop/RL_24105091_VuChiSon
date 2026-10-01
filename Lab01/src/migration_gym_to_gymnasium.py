import gymnasium as gym

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)
for t in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# terminated: episode kết thúc theo bản chất của bài toán.
# truncated: episode kết thúc do giới hạn bên ngoài.
# Không dùng done của API Gym cũ vì Gymnasium tách terminated và truncated.

env.close()