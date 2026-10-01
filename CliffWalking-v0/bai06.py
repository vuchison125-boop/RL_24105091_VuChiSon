import gymnasium as gym
import numpy as np

def value_iteration(env, gamma=0.9, theta=1e-6):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    V = np.zeros(n_states)

    while True:
        delta = 0

        for state in range(n_states):
            action_values = []

            for action in range(n_actions):
                value = 0

                for probability, next_state, reward, terminated in env.unwrapped.P[state][action]:
                    if terminated:
                        value += probability * reward
                    else:
                        value += probability * (reward + gamma * V[next_state])

                action_values.append(value)

            best_value = max(action_values)
            delta = max(delta, abs(best_value - V[state]))
            V[state] = best_value

        if delta < theta:
            break

    return V


def extract_policy(env, V, gamma):
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
                    value += probability * (reward + gamma * V[next_state])

            action_values.append(value)

        policy[state] = np.argmax(action_values)

    return policy


arrows = {
    0: "↑",
    1: "→",
    2: "↓",
    3: "←"
}

env = gym.make("CliffWalking-v1")

for gamma in [0.9, 0.99]:
    V = value_iteration(env, gamma)
    policy = extract_policy(env, V, gamma)

    print(f"\nGamma = {gamma}")
    for row in range(4):
        for col in range(12):
            state = row * 12 + col
            print(arrows[policy[state]], end=" ")
        print()

    print("Mean Value:", np.mean(V))

env.close()