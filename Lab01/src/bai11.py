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
    episode_length = step + 1
    return total_reward, episode_length

env = gym.make("CartPole-v1")
total_reward, episode_length = random_agent(env)
print("Total reward:", total_reward)
print("Episode length:", episode_length)

env.close()