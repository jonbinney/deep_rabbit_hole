from agents.adapters.base import BaseTrainableAgentAdapter
from agents.adapters.dict_split_board_adapter import DictSplitBoardAdapter
from agents.adapters.factory import create_agent_with_adapters
from agents.adapters.remove_turn_adapter import RemoveTurnAdapter
from agents.adapters.rotate_adapter import RotateAdapter
from agents.adapters.uni_board_adapter import UnifiedBoardAdapter
from agents.adapters.use_opponent_after_action_obs import UseOpponentsAfterActionObsAdapter
from agents.core.adaptable_agent import AdaptableAgent


class NDexpAgent(BaseTrainableAgentAdapter):
    @classmethod
    def params_class(cls):
        return AdaptableAgent.params_class()

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "ndexp"

    def __init__(
        self,
        observation_space,
        action_space,
        **kwargs,
    ):
        agent = create_agent_with_adapters(
            [RemoveTurnAdapter, UseOpponentsAfterActionObsAdapter, DictSplitBoardAdapter, RotateAdapter],
            AdaptableAgent,
            observation_space,
            action_space,
            **kwargs,
        )
        super().__init__(agent, **kwargs)


class CnnAgent(BaseTrainableAgentAdapter):
    @classmethod
    def params_class(cls):
        return AdaptableAgent.params_class()

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "cnn"

    def __init__(
        self,
        observation_space,
        action_space,
        **kwargs,
    ):
        agent = create_agent_with_adapters(
            [RemoveTurnAdapter, UseOpponentsAfterActionObsAdapter, RotateAdapter, UnifiedBoardAdapter],
            AdaptableAgent,
            observation_space,
            action_space,
            **kwargs,
        )
        super().__init__(agent, **kwargs)
