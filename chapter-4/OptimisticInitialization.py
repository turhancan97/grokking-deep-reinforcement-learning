import sys

sys.path.append("../")
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from enviroments.BanditWalk import SlipperyBanditWalk

# Create and use the environment
environment = gym.make("BanditSlipperyWalk-v0")
observation, info = environment.reset(seed=0)

def optimistic_initialization(env, 
                              optimistic_estimate=1.0,
                              initial_count=100,
                              n_episodes=10):
    Q = np.full((env.action_space.n), optimistic_estimate, dtype=np.float64)
    N = np.full((env.action_space.n), initial_count, dtype=np.int64)
    
    Qe = np.empty((n_episodes, env.action_space.n), dtype=np.float64)
    returns = np.empty(n_episodes, dtype=np.float64)
    actions = np.empty(n_episodes, dtype=np.int64)
    name = 'Optimistic {}, {}'.format(optimistic_estimate, 
                                     initial_count)
    for e in tqdm(range(n_episodes), desc='Episodes for: ' + name, leave=False):
        action = np.argmax(Q)

        observation, reward, terminated, truncated, info = env.step(action)
        N[action] += 1
        Q[action] = Q[action] + (reward - Q[action])/N[action]

        Qe[e] = Q
        returns[e] = reward
        actions[e] = action
    return name, returns, Qe, actions

name, returns, Qe, actions = optimistic_initialization(environment)
print(Qe)
