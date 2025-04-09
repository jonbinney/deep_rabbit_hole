from dataclasses import dataclass
from typing import Optional

from utils import SubargsBase

from agents.core import Agent


@dataclass
class HumanParams(SubargsBase):
    nick: Optional[str] = None


class HumanAgent(Agent):
    def __init__(self, params=HumanParams(), **kwargs):
        super().__init__()
        self.params = params

    @classmethod
    def params_class(cls):
        return HumanParams

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "Human"

    def get_action(self, game):
        # Import this here to avoid circular dependencies
        from renderers.pygame import PygameQuoridor

        observation, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        # Transform the action mask into a set of valid moves
        valid_moves = set()
        for action, value in enumerate(observation["action_mask"]):
            if value == 1:
                r, c, type = game.action_index_to_params(action)
                valid_moves.add((r, c, type))

        result = PygameQuoridor.instance().get_human_input(valid_moves)

        return None if result is None else game.action_params_to_index(*result)
