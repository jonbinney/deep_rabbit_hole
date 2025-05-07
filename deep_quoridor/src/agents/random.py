from agents.core import Agent


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

        self.action_space = kwargs["action_space"]

    def get_action(self, observation):
        action_mask = observation["action_mask"]
        return self.action_space.sample(action_mask)
