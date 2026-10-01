import gymnasium as gym

def policy(observation):
    return 0 if observation is None else env.action_space.sample()

def run_episode(env):
    observation, info = env.reset()
    total_reward = 0
    for _ in range(500):
        action = policy(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    return total_reward

env = gym.make("CartPole-v1")
reward = run_episode(env)
print("Total reward:", reward)
env.close()