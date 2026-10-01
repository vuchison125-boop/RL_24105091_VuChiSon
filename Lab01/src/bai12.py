import gymnasium as gym

def random_agent(env, max_steps=500):
    observation, info = env.reset()
    total_reward = 0.0
    for step in range(max_steps):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_finished = terminated or truncated
        if episode_finished:
            break
    episode_length = step + 1
    return total_reward, episode_length, terminated, truncated


env = gym.make("CartPole-v1")
total_reward, episode_length, terminated, truncated = random_agent(env)
print("Total reward:", total_reward)
print("Episode length:", episode_length)

if terminated:
    print("Termination")
elif truncated:
    print("Truncation")

env.close()