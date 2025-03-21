from agents import SelfRegisteringAgent


class RandomAgent(SelfRegisteringAgent):
    def __init__(self):
        super().__init__()

    def get_action(self, game):
        observation, _, termination, truncation, _ = game.last()
        mask = observation["action_mask"]
        if termination or truncation:
            return None
        return game.action_space(game.agent_selection).sample(mask)
