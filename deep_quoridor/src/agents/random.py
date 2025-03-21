from agents import Agent


class RandomAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

    def get_action(self, game):
        observation, _, termination, truncation, _ = game.last()
        mask = observation["action_mask"]
        if termination or truncation:
            return None
        return game.action_space(game.agent_selection).sample(mask)
