import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

def policy_evaluation(pi, P, gamma=0.99, theta=1e-10):
    """
    Evaluate the value function for a given policy using iterative policy evaluation.

    Parameters:
    - pi (list): The policy, where pi[s] is the action to take in state s.
    - P (list): The transition probabilities and rewards for each state-action pair.
    - gamma (float): Discount factor for future rewards (default: 0.99).
    - theta (float): Threshold for convergence (default: 1e-10).

    Returns:
    - v (list): The value function, where v[s] is the estimated value of state s.
    - count (int): The number of iterations until convergence.
    """
    prev_v = np.zeros(len(P))
    count = 0
    while True:
        count += 1
        v = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_s, reward, done in P[s][pi[s]]:
                v[s] += prob * (reward + gamma * prev_v[next_s] * (not done))
        if np.max(np.abs(prev_v - v)) < theta:
            break
        prev_v = v.copy()
    return v, count

# create a random policy for the 4x4 grid
# 0: Move left
# 1: Move down
# 2: Move right
# 3: Move up

# Go-get-it policy
pi_1 = np.array([2, 2, 1, 0,
               1, 0, 1, 0,
               2, 2, 1, 0,
               0, 2, 2, 0])

# The Careful policy
pi_2 = np.array([0, 3, 3, 3,
               0, 0, 3, 0,
               3, 1, 0, 0,
               0, 2, 2, 0])

# Random policy
pi_random = np.random.choice(4, 16)

P = env.unwrapped.P  # transition probabilities (MDP)
v, count = policy_evaluation(pi_2, P)  # policy evaluation

print_ = False
if print_:
    print("Number of iterations: ", count)
    print(v.reshape(4, 4))
