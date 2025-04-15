from dataclasses import dataclass
from typing import Optional

from quoridor import ActionEncoder
from utils import SubargsBase

from agents.core import Agent


@dataclass
class HumanParams(SubargsBase):
    nick: Optional[str] = None


class HumanAgent(Agent):
    def __init__(self, params=HumanParams(), **kwargs):
        super().__init__()
        self.params = params
        self.action_encoder = ActionEncoder(kwargs["board_size"])

    @classmethod
    def params_class(cls):
        return HumanParams

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "Human"

    def get_action(self, observation, action_mask):
        # Import this here to avoid circular dependencies
        from renderers.pygame import PygameQuoridor

        # Transform the action mask into a set of valid moves
        valid_moves = set()
        for action, value in enumerate(action_mask):
            if value == 1:
                valid_action = self.action_encoder.index_to_action(action)
                valid_moves.add(valid_action)

        result = PygameQuoridor.instance().get_human_input(valid_moves)

        return None if result is None else self.action_encoder.action_to_index(*result)
