import gymnasium as gym
import numpy as np

def improved_policy(observation):
    pole_angle = observation[2]
    pole_angular_velocity = observation[3]
    score = pole_angle + 0.5 * pole_angular_velocity
    if score < 0:
        return 0
    else:
        return 1

def random_policy(observation, env):
    return env.action_space.sample()

def evaluate_policy(env, policy, n_episodes=100, random=False):
    rewards = []
    for _ in range(n_episodes):
        observation, info = env.reset()
        total_reward = 0
        for _ in range(500):
            if random:
                action = random_policy(observation, env)
            else:
                action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

env = gym.make("CartPole-v1")

improved_mean = evaluate_policy(env, improved_policy)
random_mean = evaluate_policy(env, improved_policy, random=True)
print(f"Improved policy mean reward: {improved_mean:.2f}")
print(f"Random policy mean reward: {random_mean:.2f}")

env.close()