import gymnasium as gym
import numpy as np
env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

def value_iteration(P, gamma=0.99, theta=1e-10):

    v = np.zeros(len(P), dtype=np.float64)

    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_s, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * v[next_s] * (not done))

        if np.max(np.abs(v - np.max(Q, axis=1))) < theta:
            break

        v = np.max(Q, axis=1)

    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return v, pi

P = env.unwrapped.P  # transition probabilities (MDP)
v, pi = value_iteration(P)  # value iteration

print_ = True
if print_:
    print('Value Function: ')
    print(v.reshape(4, 4))
    list_pi = np.array([pi(i) for i in range(16)])
    print()
    print('Policy:')
    print(list_pi.reshape(4, 4))
