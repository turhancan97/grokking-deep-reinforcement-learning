import sys

sys.path.append("../")
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from enviroments.BanditWalk import SlipperyBanditWalk

# Create and use the environment
environment = gym.make("BanditSlipperyWalk-v0")
observation, info = environment.reset()
print("observation: ", observation)
print("info: ", info)


def pure_exploration(env, n_episodes=10):
    Q = np.zeros((env.action_space.n), dtype=np.float64)
    N = np.zeros((env.action_space.n), dtype=np.int64)

    Qe = np.empty((n_episodes, env.action_space.n), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int64)
    name = "Pure exploration"
    for e in tqdm(range(n_episodes), desc="Episodes for: " + name, leave=False):
        action = np.random.randint(len(Q))

        observation, reward, terminated, truncated, info = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action]) / N[action]

        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions

name, returns, Qe, actions = pure_exploration(environment)
