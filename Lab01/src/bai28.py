import gymnasium as gym
import numpy as np

def run_experiment(is_slippery, n_episodes=500):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    rewards = []
    lengths = []
    success = 0
    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0
        for step in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                if reward == 1:
                    success += 1
                break
        rewards.append(total_reward)
        lengths.append(step + 1)
    env.close()
    return (
        success / n_episodes,
        np.mean(rewards),
        np.mean(lengths)
    )

deterministic = run_experiment(False)
stochastic = run_experiment(True)

print("is_slippery=False")
print(f"Success rate: {deterministic[0]:.2f}")
print(f"Average reward: {deterministic[1]:.2f}")
print(f"Average episode length: {deterministic[2]:.2f}")
print()
print("is_slippery=True")
print(f"Success rate: {stochastic[0]:.2f}")
print(f"Average reward: {stochastic[1]:.2f}")
print(f"Average episode length: {stochastic[2]:.2f}")

# Khi môi trường stochastic, action có thể không tạo ra trạng thái mong muốn.
# Vì vậy hiệu quả của random policy có thể thay đổi so với môi trường deterministic.