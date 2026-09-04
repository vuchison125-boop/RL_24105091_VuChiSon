import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def random_policy(observation, env):
    return env.action_space.sample()

def angle_based_policy(observation, env):
    pole_angle = observation[2]
    if pole_angle < 0:
        return 0
    else:
        return 1

def improved_policy(observation, env):
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
    return total_reward, step + 1

def evaluate_policy(env_name, policy, n_episodes=500, seed=42):
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
    result = {
        "mean_reward": np.mean(rewards),
        "std": np.std(rewards),
        "min": np.min(rewards),
        "max": np.max(rewards),
        "mean_length": np.mean(lengths)
    }
    env.close()
    return result

results = {
    "Random": evaluate_policy("CartPole-v1", random_policy),
    "Angle-based": evaluate_policy("CartPole-v1", angle_based_policy),
    "Improved": evaluate_policy("CartPole-v1", improved_policy)
}

print("Agent | Mean reward | Std | Min | Max | Mean length")

for agent, result in results.items():
    print(
        f"{agent} | "
        f"{result['mean_reward']:.2f} | "
        f"{result['std']:.2f} | "
        f"{result['min']:.2f} | "
        f"{result['max']:.2f} | "
        f"{result['mean_length']:.2f}"
    )

agents = list(results.keys())
mean_rewards = [
    results[agent]["mean_reward"]
    for agent in agents
]

plt.bar(agents, mean_rewards)
plt.title("Comparison of Agents")
plt.xlabel("Agent")
plt.ylabel("Mean Reward")
plt.grid(axis="y")
plt.savefig("figures/comparison_agents.png")
plt.show()