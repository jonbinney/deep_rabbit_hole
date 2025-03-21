class Agent:
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    agents = {}

    def __init_subclass__(cls, **kwargs):
        friendly_name = Agent._friendly_name(cls.__name__)
        Agent.agents[friendly_name] = cls

    def name(self):
        return Agent._friendly_name(self.__class__.__name__)

    @staticmethod
    def _friendly_name(class_name: str):
        return class_name.replace("Agent", "").lower()

    @staticmethod
    def create(friendly_name: str) -> "Agent":
        return Agent.agents[friendly_name]()

    @staticmethod
    def names():
        return list(Agent.agents.keys())

    def get_action(self, game):
        raise NotImplementedError("You must implement the get_action method")


__all__ = ["RandomAgent", "SimpleAgent", "Agent", "DQNAgent"]

from agents.random import RandomAgent  # noqa: E402
from agents.simple import SimpleAgent  # noqa: E402
from agents.dqn import DQNAgent  # noqa: E402
