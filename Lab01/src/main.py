import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def random_policy(observation, env):
    return env.action_space.sample()

def run_episode(env, policy, seed=None, max_steps=1000):
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)
    total_reward = 0
    for step in range(max_steps):
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, step + 1

def evaluate_policy(
    env_name,
    policy,
    n_episodes=100,
    seed=42
):
    env = gym.make(env_name)
    rewards = []
    lengths = []
    for episode in range(n_episodes):
        reward, length = run_episode(
            env,
            policy,
            seed=seed + episode
        )
        rewards.append(reward)
        lengths.append(length)
    env.close()
    return rewards, lengths


def plot_rewards(rewards):
    episodes = np.arange(1, len(rewards) + 1)
    plt.plot(episodes, rewards)
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.show()


def main():
    rewards, lengths = evaluate_policy(
        "CartPole-v1",
        random_policy,
        n_episodes=100,
        seed=42
    )
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std reward: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.2f}")
    print(f"Max reward: {np.max(rewards):.2f}")
    plot_rewards(rewards)

if __name__ == "__main__":
    main()