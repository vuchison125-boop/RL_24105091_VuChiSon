import gymnasium as gym

def run_episode(env, policy, seed=None, max_steps=1000):
    if seed is None:
        observation, info = env.reset()
    else:
        observation, info = env.reset(seed=seed)
    total_reward = 0
    for step in range(max_steps):
        action = policy(observation, env)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return {
        "reward": total_reward,
        "length": step + 1,
        "terminated": terminated,
        "truncated": truncated
    }

def random_policy(observation, env):
    return env.action_space.sample()

env = gym.make("CartPole-v1")
result = run_episode(env, random_policy, seed=42)
print(result)

env.close()