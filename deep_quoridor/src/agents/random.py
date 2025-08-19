import numpy as np

from agents.core import Agent


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

        self.action_space = kwargs["action_space"]

    def get_action(self, observation):
        action_mask = observation["action_mask"]
        # The action mask has dtype=float32 by default to make it easy to use with pytorch, but sampling
        # the action space requires int8.
        return self.action_space.sample(np.astype(action_mask, np.int8))
