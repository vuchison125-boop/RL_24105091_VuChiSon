import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)
total_episodes = 100
success = 0
failure = 0

for _ in range(total_episodes):
    observation, info = env.reset()
    for _ in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            if reward == 1:
                success += 1
            else:
                failure += 1
            break

success_rate = success / total_episodes

print("Total episodes:", total_episodes)
print("Success:", success)
print("Failure:", failure)
print(f"Success rate: {success_rate:.2f}")

env.close()