import gymnasium as gym

def random_agent(env, max_steps=500):
    observation, info = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward, step + 1

env = gym.make("CartPole-v1")
print("Episode | Reward | Length")
for episode in range(1, 11):
    reward, length = random_agent(env)
    print(f"{episode:7d} | {reward:6.1f} | {length:6d}")

env.close()