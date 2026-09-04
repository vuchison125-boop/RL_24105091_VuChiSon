import gymnasium as gym


def run_one_step(env, action):
    observation, reward, terminated, truncated, info = env.step(action)
    return observation, reward, terminated, truncated, info

env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for action in [0, 1, 0, 1, 0]:
    result = run_one_step(env, action)
    print("Action:", action)
    print("Result:", result)

env.close()