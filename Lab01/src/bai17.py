import gymnasium as gym
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

env = gym.make("CartPole-v1")
episode_rewards = []
for _ in range(100):
    reward = random_agent(env)
    episode_rewards.append(reward)
episodes = range(1, 101)

plt.plot(episodes, episode_rewards)
plt.title("Reward per Episode - CartPole")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid()
plt.savefig("figures/reward_cartpole.png")
plt.show()

env.close()