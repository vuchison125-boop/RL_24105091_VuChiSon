import gymnasium as gym

env = gym.make("CartPole-v1")
env.action_space.seed(42)
actions = []

for _ in range(20):
    actions.append(env.action_space.sample())

print("Actions:", actions)

env.close()