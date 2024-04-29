import sys

sys.path.append("../")

import numpy as np
import random
from tqdm import tqdm
import gymnasium as gym
from utils import decay_schedule,generate_trajectory, plot_value_function, policy_evaluation, rmse
from utils import print_state_value_function, print_policy, probability_success, mean_return
import gym_walk
import matplotlib.pyplot as plt
SEEDS = (12, 34, 56, 78, 90)

environment = gym.make("RandomWalk-v0")
init_state, info = environment.reset(seed=42)
goal_state = 6
gamma = 1.0
n_episodes = 500
P = environment.env.P

LEFT, RIGHT = range(2)
pi = lambda s: {0: LEFT, 1: LEFT, 2: LEFT, 3: LEFT, 4: LEFT, 5: LEFT, 6: LEFT}[s]
V_true = policy_evaluation(pi, P, gamma=gamma)
print_state_value_function(V_true, P, n_cols=7)
print()
print_policy(pi, P, action_symbols=('<', '>'), n_cols=7)
print('Reaches goal {:.2f}%. Obtains an average return of {:.4f}.'.format(
    probability_success(environment, pi, goal_state=goal_state), 
    mean_return(environment, gamma, pi)))

def mc_prediction(
    pi,
    env,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    n_episodes=500,
    max_steps=200,
    first_visit=True,
):
    nS = env.observation_space.n
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    V = np.zeros(nS, dtype=np.float64)
    V_track = np.zeros((n_episodes, nS), dtype=np.float64)
    targets = {state: [] for state in range(nS)}

    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(pi, env, max_steps)
        visited = np.zeros(nS, dtype=bool)
        for t, (state, _, reward, _, _) in enumerate(trajectory):
            if visited[state] and first_visit:
                continue
            visited[state] = True

            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            targets[state].append(G)
            mc_error = G - V[state]
            V[state] = V[state] + alphas[e] * mc_error
        V_track[e] = V
    return V.copy(), V_track, targets


V_fvmc, V_track_fvmc, targets_fvmc = mc_prediction(
    pi, environment, gamma=gamma, n_episodes=n_episodes
)

V_fvmcs, V_track_fvmcs = [], []
for seed in tqdm(SEEDS, desc="All seeds", leave=True):
    random.seed(seed)
    np.random.seed(seed)
    environment.reset(seed=seed)
    V_fvmc, V_track_fvmc, targets_fvmc = mc_prediction(
        pi, environment, gamma=gamma, n_episodes=n_episodes
    )
    V_fvmcs.append(V_fvmc)
    V_track_fvmcs.append(V_track_fvmc)
V_fvmc, V_track_fvmc = np.mean(V_fvmcs, axis=0), np.mean(V_track_fvmcs, axis=0)

del V_fvmcs
del V_track_fvmcs

print_state_value_function(V_fvmc, P, n_cols=7)
print()
print_state_value_function(V_fvmc - V_true, P, n_cols=7, title='State-value function errors:')
print('RMSE:', rmse(V_fvmc, V_true))

plot_value_function(
    "FVMC estimates through time vs. true values", V_track_fvmc, V_true, log=False
)

V_evmcs, V_track_evmcs = [], []
for seed in tqdm(SEEDS, desc="All seeds", leave=True):
    random.seed(seed)
    np.random.seed(seed)
    environment.reset(seed=seed)
    V_evmc, V_track_evmc, targets_evmc = mc_prediction(
        pi, environment, gamma=gamma, n_episodes=n_episodes, first_visit=False
    )
    V_evmcs.append(V_evmc)
    V_track_evmcs.append(V_track_evmc)
V_evmc, V_track_evmc = np.mean(V_evmcs, axis=0), np.mean(V_track_evmcs, axis=0)

del V_evmcs
del V_track_evmcs

print_state_value_function(V_evmc, P, n_cols=7)
print()
print_state_value_function(V_evmc - V_true, P, n_cols=7, title='State-value function errors:')
print('RMSE:', rmse(V_evmc, V_true))

plot_value_function(
    "EVMC estimates through time vs. true values", V_track_evmc, V_true, log=False
)
