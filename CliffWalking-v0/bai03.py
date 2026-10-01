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
                        value += probability * (
                            reward + gamma * V[next_state]
                        )
                action_values.append(value)
            best_value = max(action_values)
            delta = max(
                delta,
                abs(best_value - V[state])
            )
            V[state] = best_value
        if delta < theta:
            break
    return V

env = gym.make("CliffWalking-v1")
V = value_iteration(env)

print("Value Function:")
print(V)
print()
print("Mean Value:", np.mean(V))

env.close()