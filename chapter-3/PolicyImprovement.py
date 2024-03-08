import gymnasium as gym
import numpy as np
from PolicyEvaluation import policy_evaluation, P

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

def policy_improvement(V, P, gamma=0.99):
    """
    Performs policy improvement given the value function and the transition probabilities.

    Parameters:
        V (numpy.ndarray): The value function for each state.
        P (list): The transition probabilities for each state-action pair.
        gamma (float, optional): The discount factor. Defaults to 0.99.

    Returns:
        tuple: A tuple containing the new policy function and the Q-values.

    """
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_s, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_s] * (not done))

    new_pi = lambda s: {
        s: a for s, a in enumerate(np.argmax(Q, axis=1))
    }[s]

    return new_pi, Q

# 0: Move left
# 1: Move down
# 2: Move right
# 3: Move up

# The Careful+ policy
pi = np.array([0, 3, 3, 3,
               0, 0, 3, 0,
               3, 1, 0, 0,
               0, 2, 2, 0])

v, count = policy_evaluation(pi, P)  # policy evaluation

new_pi, Q = policy_improvement(v, P)  # policy improvement

print_ = False
if print_:
    list_pi = np.array([new_pi(i) for i in range(16)])
    print()
    print('New policy:')
    print(list_pi.reshape(4, 4))

    env_dict = dict()
    for i in range(4):
        for j in range(4):
            env_dict[(i, j)] = Q[j]
    print()
    print('Q values:')
    print(env_dict)
