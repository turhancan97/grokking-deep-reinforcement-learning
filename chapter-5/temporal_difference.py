import sys

sys.path.append("../")

import numpy as np
import random
from tqdm import tqdm
import gymnasium as gym
from utils import (
    decay_schedule,
    generate_trajectory,
    plot_value_function,
    policy_evaluation,
    rmse,
)
from utils import (
    print_state_value_function,
    print_policy,
    probability_success,
    mean_return,
)
import gym_walk
import matplotlib.pyplot as plt

SEEDS = (12, 34, 56, 78, 90)

environment = gym.make("RandomWalk-v0")
init_state, info = environment.reset(seed=42)
goal_state = 6
gamma = 1.0
n_episodes = 500
P = environment.unwrapped.P

LEFT, RIGHT = range(2)
pi = lambda s: {0: LEFT, 1: LEFT, 2: LEFT, 3: LEFT, 4: LEFT, 5: LEFT, 6: LEFT}[s]
V_true = policy_evaluation(pi, P, gamma=gamma)
print_state_value_function(V_true, P, n_cols=7)
print()
print_policy(pi, P, action_symbols=("<", ">"), n_cols=7)
print(
    "Reaches goal {:.2f}%. Obtains an average return of {:.4f}.".format(
        probability_success(environment, pi, goal_state=goal_state),
        mean_return(environment, gamma, pi),
    )
)


def td(
    pi,
    env,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    n_episodes=500,
):
    nS = env.observation_space.n
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {state: [] for state in range(nS)}
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
        state, info = env.reset()
        done = False
        while not done:
            action = pi(state)
            next_state, reward, done, _, _ = env.step(action)
            td_target = reward + gamma * V[next_state] * (not done)
            targets[state].append(td_target)
            td_error = td_target - V[state]
            V[state] = V[state] + alphas[e] * td_error
            state = next_state
        V_track[e] = V
    return V, V_track, targets


V_tds, V_track_tds = [], []
for seed in tqdm(SEEDS, desc="All seeds", leave=True):
    random.seed(seed)
    np.random.seed(seed)
    environment.reset(seed=seed)
    V_td, V_track_td, targets_td = td(pi, environment, gamma=gamma, n_episodes=n_episodes)
    V_tds.append(V_td)
    V_track_tds.append(V_track_td)
V_td, V_track_td = np.mean(V_tds, axis=0), np.mean(V_track_tds, axis=0)
del V_tds
del V_track_tds

print_state_value_function(V_td, P, n_cols=7)
print()
print_state_value_function(
    V_td - V_true, P, n_cols=7, title="State-value function errors:"
)
print("RMSE:", rmse(V_td, V_true))

plot_value_function(
    "TD estimates through time vs. true values", V_track_td, V_true, log=False
)
