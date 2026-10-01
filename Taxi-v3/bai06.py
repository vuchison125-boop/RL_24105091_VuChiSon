import gymnasium as gym
import numpy as np


def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    V = np.zeros(n_states)

    while True:
        delta = 0

        for state in range(n_states):
            action = policy[state]
            new_value = 0

            for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                if terminated:
                    new_value += probability * reward
                else:
                    new_value += probability * (
                        reward + gamma * V[next_state]
                    )

            delta = max(delta, abs(new_value - V[state]))
            V[state] = new_value

        if delta < theta:
            break

    return V


def policy_improvement(env, V, gamma=0.9):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    policy = np.zeros(n_states, dtype=int)

    for state in range(n_states):
        action_values = []

        for action in range(n_actions):
            value = 0

            for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                if terminated:
                    value += probability * reward
                else:
                    value += probability * (
                        reward + gamma * V[next_state]
                    )

            action_values.append(value)

        policy[state] = np.argmax(action_values)

    return policy


def policy_iteration(env, gamma=0.9):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    np.random.seed(42)
    policy = np.random.randint(n_actions, size=n_states)

    iterations = 0

    while True:
        V = policy_evaluation(env, policy, gamma)
        new_policy = policy_improvement(env, V, gamma)

        iterations += 1

        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return policy, V, iterations


env = gym.make("Taxi-v4")

policy, V, iterations = policy_iteration(env)

rewards = []
steps = []
successes = 0

for episode in range(1000):
    state, info = env.reset(seed=episode)

    total_reward = 0
    step_count = 0

    while True:
        action = policy[state]

        state, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        if terminated or truncated:
            break

    rewards.append(total_reward)
    steps.append(step_count)

    if reward == 20:
        successes += 1

print("Iterations:", iterations)
print("Mean Value:", np.mean(V))
print("Average Reward:", np.mean(rewards))
print("Average Steps:", np.mean(steps))
print("Successful Episodes:", successes)
print("Success Rate:", successes / 1000)

env.close()