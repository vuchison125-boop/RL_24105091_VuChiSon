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


def extract_policy(env, V, gamma=0.9):
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


for cliff_reward in [-100, -10]:
    env = gym.make("CliffWalking-v1")

    # Thay đổi reward của các trạng thái cliff
    for state in range(37, 47):
        for action in range(4):
            transitions = env.unwrapped.P[state][action]

            for i, transition in enumerate(transitions):
                probability, next_state, reward, terminated = transition

                if next_state == 36 and terminated:
                    transitions[i] = (
                        probability,
                        next_state,
                        cliff_reward,
                        terminated
                    )

    V = value_iteration(env)
    policy = extract_policy(env, V)

    print(f"\nCliff Reward = {cliff_reward}")

    for row in range(4):
        for col in range(12):
            state = row * 12 + col
            print(arrows[policy[state]], end=" ")
        print()

    print("Mean Value:", np.mean(V))

    env.close()