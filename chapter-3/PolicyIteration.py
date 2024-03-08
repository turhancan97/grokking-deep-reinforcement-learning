import gymnasium as gym
import numpy as np
from PolicyEvaluation import policy_evaluation
from PolicyImprovement import policy_improvement
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

def policy_iteration(P, gamma=0.99, theta=1e-10):

    random_actions = np.random.choice(tuple(P[0].keys()), len(P))

    pi = lambda s: {s: a for s, a in enumerate(random_actions)}[s]

    while True:
        old_pi = {s: pi(s) for s in range(len(P))}
        list_pi = np.array([pi(i) for i in range(len(P))])
        v, _ = policy_evaluation(list_pi, P, gamma, theta)
        pi, _ = policy_improvement(v, P, gamma)

        if old_pi == {s: pi(s) for s in range(len(P))}:
            break

    return v, pi


P = env.unwrapped.P  # transition probabilities (MDP)
v, pi = policy_iteration(P)  # policy iteration
print_ = False
if print_:
    print('Value Function: ')
    print(v.reshape(4, 4))
    if print_:
        list_pi = np.array([pi(i) for i in range(16)])
        print()
        print('New policy:')
        print(list_pi.reshape(4, 4))
