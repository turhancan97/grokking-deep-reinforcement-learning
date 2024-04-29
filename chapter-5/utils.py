import numpy as np
import random
import itertools
from tabulate import tabulate
from collections import defaultdict
from itertools import cycle, count
import matplotlib.pyplot as plt


def decay_schedule(
    init_value, min_value, decay_ratio, max_steps, log_start=-2, log_base=10
):
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    values = np.logspace(log_start, 0, decay_steps, base=log_base, endpoint=True)[::-1]
    values = (values - values.min()) / (values.max() - values.min())
    values = (init_value - min_value) * values + min_value
    values = np.pad(values, (0, rem_steps), "edge")
    return values


# plt.plot(decay_schedule(0.5, 0.01, 0.5, 500))
# plt.title('Exponentially decaying schedule (for alpha)')
# plt.xticks(rotation=45)
# plt.show()


def generate_trajectory(pi, env, max_steps=200):
    done, trajectory = False, []
    while not done:
        state, _ = env.reset()
        for t in count():
            action = pi(state)
            next_state, reward, done, _, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done:
                break
            if t >= max_steps - 1:
                trajectory = []
                break
            state = next_state
    return np.array(trajectory, dtype=object)


def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P), dtype=np.float64)
    while True:
        V = np.zeros(len(P), dtype=np.float64)
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V


def rmse(x, y, dp=4):
    return np.round(np.sqrt(np.mean((x - y) ** 2)), dp)


def print_policy(pi, P, action_symbols=("<", "v", ">", "^"), n_cols=4, title="Policy:"):
    print(title)
    arrs = {k: v for k, v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0:
            print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title="State-value function:"):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), "{}".format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0:
            print("|")


def print_action_value_function(
    Q, optimal_Q=None, action_symbols=("<", ">"), prec=3, title="Action-value function:"
):
    vf_types = ("",) if optimal_Q is None else ("", "*", "er")
    headers = [
        "s",
    ] + [" ".join(i) for i in list(itertools.product(vf_types, action_symbols))]
    print(title)
    states = np.arange(len(Q))[..., np.newaxis]
    arr = np.hstack((states, np.round(Q, prec)))
    if not (optimal_Q is None):
        arr = np.hstack((arr, np.round(optimal_Q, prec), np.round(optimal_Q - Q, prec)))
    print(tabulate(arr, headers, tablefmt="fancy_grid"))


def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.reset(seed=123)
    results = []
    for _ in range(n_episodes):
        state, info = env.reset()
        done, steps = False, 0
        while not done and steps < max_steps:
            state, _, done, h, _ = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results) / len(results) * 100


def mean_return(env, gamma, pi, n_episodes=100, max_steps=200):
    random.seed(123)
    np.random.seed(123)
    env.reset(seed=123)
    results = []
    for _ in range(n_episodes):
        state, info = env.reset()
        done, steps = False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _, _ = env.step(pi(state))
            results[-1] += gamma**steps * reward
            steps += 1
    return np.mean(results)


def plot_value_function(
    title, V_track, V_true=None, log=False, limit_value=0.05, limit_items=5
):
    np.random.seed(123)
    per_col = 25
    linecycler = cycle(["-", "--", ":", "-."])
    legends = []

    valid_values = np.argwhere(V_track[-1] > limit_value).squeeze()
    items_idxs = np.random.choice(
        valid_values, min(len(valid_values), limit_items), replace=False
    )
    # draw the true values first
    if V_true is not None:
        for i, state in enumerate(V_track.T):
            if i not in items_idxs:
                continue
            if state[-1] < limit_value:
                continue

            label = "v({})".format(i)
            plt.axhline(y=V_true[i], color="k", linestyle="-", linewidth=1)
            plt.text(int(len(V_track) * 1.02), V_true[i] + 0.01, label)

    # then the estimates
    for i, state in enumerate(V_track.T):
        if i not in items_idxs:
            continue
        if state[-1] < limit_value:
            continue
        line_type = next(linecycler)
        label = "V({})".format(i)
        (p,) = plt.plot(state, line_type, label=label, linewidth=3)
        legends.append(p)

    legends.reverse()

    ls = []
    for loc, idx in enumerate(range(0, len(legends), per_col)):
        subset = legends[idx : idx + per_col]
        l = plt.legend(subset, [p.get_label() for p in subset])
        ls.append(l)
    [plt.gca().add_artist(l) for l in ls[:-1]]
    if log:
        plt.xscale("log")
    plt.title(title)
    plt.ylabel("State-value function")
    plt.xlabel("Episodes (log scale)" if log else "Episodes")
    plt.show()
