import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def create_environment():
    return gym.make("CartPole-v1")

def policy(observation, env):
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    score = pole_angle + 0.5 * pole_angular_velocity
    if score < 0:
        return 0
    else:
        return 1

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
    return {
        "reward": total_reward,
        "length": step + 1,
        "terminated": terminated,
        "truncated": truncated
    }

def evaluate_policy(env, policy, n_episodes=500, seed=42):
    rewards = []
    lengths = []
    for episode in range(n_episodes):
        result = run_episode(
            env,
            policy,
            seed=seed + episode
        )
        rewards.append(result["reward"])
        lengths.append(result["length"])
    return rewards, lengths

def plot_results(rewards):
    episodes = np.arange(1, len(rewards) + 1)
    window_size = 10
    moving_avg = np.convolve(
        rewards,
        np.ones(window_size) / window_size,
        mode="valid"
    )
    moving_avg_episodes = np.arange(
        window_size,
        len(rewards) + 1
    )
    plt.figure()
    plt.plot(episodes, rewards)
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig("figures/bai36_reward.png")
    plt.show()
    plt.figure()
    plt.plot(
        moving_avg_episodes,
        moving_avg
    )
    plt.title("Moving Average")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid()
    plt.savefig("figures/bai36_moving_average.png")
    plt.show()

def main():
    env = create_environment()
    rewards, lengths = evaluate_policy(
        env,
        policy,
        n_episodes=500,
        seed=42
    )
    rewards = np.array(rewards)
    lengths = np.array(lengths)
    best_index = np.argmax(rewards)
    worst_index = np.argmin(rewards)

    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Standard deviation: {np.std(rewards):.2f}")
    print(f"Best episode: {best_index + 1}")
    print(f"Best reward: {rewards[best_index]:.2f}")
    print(f"Worst episode: {worst_index + 1}")
    print(f"Worst reward: {rewards[worst_index]:.2f}")
    plot_results(rewards)
    env.close()

if __name__ == "__main__":
    main()