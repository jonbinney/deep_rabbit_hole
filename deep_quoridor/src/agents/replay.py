from agents.core import Agent


class ReplayAgent(Agent):
    """A replay agent that plays predefined actions in sequence.

    This agent is used for replaying a sequence of actions, typically for testing or
    demonstration purposes. It simply returns actions from a predefined list in order.

    Args:
        actions (list[int]): A list of predefined actions to be played in sequence.

    Attributes:
        actions (list[int]): The list of predefined actions.
        action_index (int): Current index in the actions list.
    """

    def __init__(self, name: str, predefined_actions: list[int]):
        super().__init__()
        self.actions = predefined_actions
        self.action_index = 0
        self.original_name = name

    def get_action(self, game):
        action = self.actions[self.action_index]
        self.action_index += 1
        return action

    def name(self):
        return f"replay-{self.original_name}"
