import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def random_agent(env, max_steps=500):
    observation, info = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    return total_reward

def moving_average(values, window_size):
    values = np.array(values)

    return np.convolve(
        values,
        np.ones(window_size) / window_size,
        mode="valid"
    )

env = gym.make("CartPole-v1")
episode_rewards = []
for _ in range(100):
    reward = random_agent(env)
    episode_rewards.append(reward)

window_size = 10
moving_avg = moving_average(episode_rewards, window_size)
episodes = np.arange(1, 101)
moving_avg_episodes = np.arange(window_size, 101)

plt.plot(episodes, episode_rewards, label="Reward")
plt.plot(moving_avg_episodes, moving_avg, label="Moving Average")
plt.title("Reward and Moving Average - CartPole")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.legend()
plt.savefig("figures/moving_average.png")
plt.show()

env.close()