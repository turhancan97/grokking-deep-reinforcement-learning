import gymnasium as gym
from gymnasium import Env, spaces, register
import random


class SlipperyBanditWalk(Env):

    # ----- 1 -----
    # constructor for initialization and some helper functions

    def __init__(self, render_mode=None, slip_prob=0.2):
        # P is basically State: Action: [ Transition Probability , Next state , Reward , isTerminated?]
        self.P = {
            0: {0: [(1.0, 0, 0.0, True)], 1: [(1.0, 0, 0.0, True)]},
            1: {
                0: [(0.8, 0, 0.0, True), (0.2, 2, 1.0, True)],
                1: [(0.8, 2, 1.0, True), (0.2, 0, 0.0, True)],
            },
            2: {0: [(1.0, 2, 0.0, True)], 1: [(1.0, 2, 0.0, True)]},
        }

        self.size = 3  # The size of the 1D grid

        # We have 3 observations, corresponding to each position in the 1-D grid
        self.observation_space = spaces.Discrete(self.size)

        # We have 2 actions, corresponding to "left" & "right"
        self.action_space = spaces.Discrete(2)

        # The probability of the slip in this case 0.2
        self.slip_prob = slip_prob

    # return the locations of agent and target
    def _get_obs(self):
        return self._agent_location

    # returns the distance between agent and target
    def _get_info(self):
        return {"agent": self._agent_location, "target": self._target_location,
                "distance": abs(self._agent_location - self._target_location)}

    # ----- 2 ------
    # The reset function to initiate

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._agent_location = 1  # location of agent in the begining
        self._target_location = (
            self.size - 1
        )  # starting location of target in this case 2
        self._dead_state = 0  # dead location

        observation = self._get_obs()  # getting useful information
        info = self._get_info()

        return observation, info

    # ------- 3 ---------
    # The step function

    def step(self, action):  # takes action as a parameter

        # gets the current location and stores the values from P set
        prev_location = self._agent_location  # gets location
        transitions = self.P[prev_location][
            action
        ]  # gets the corresponding action tuple
        probabilities, next_states, rewards, terminals = zip(
            *transitions
        )  # stores the value for use

        # Randomly select a transition based on the probabilities
        # gives you random state based on your probabilities
        index = random.choices(range(len(probabilities)), weights=probabilities, k=1)[0]
        # stores the values
        self._agent_location, reward, terminated = (
            next_states[index],
            rewards[index],
            terminals[index],
        )

        truncated = False
        observation = self._get_obs()
        info = self._get_info()

        info["log"] = {
            "current_state": prev_location,
            "action": action,
            "next_state": self._agent_location,
        }

        # Return the required 5-tuple
        return observation, reward, terminated, truncated, info

# Register the custom environment
register(id="BanditSlipperyWalk-v0", entry_point=SlipperyBanditWalk)
