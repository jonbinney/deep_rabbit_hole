class Agent:
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    def name(self) -> str:
        raise NotImplementedError("You must implement the name method")

    def get_action(self, game) -> int:
        raise NotImplementedError("You must implement the get_action method")


class AgentRegistry:
    agents = {}

    @staticmethod
    def create(friendly_name: str, **kwargs) -> Agent:
        return AgentRegistry.agents[friendly_name](**kwargs)

    @staticmethod
    def names():
        return list(AgentRegistry.agents.keys())

    @staticmethod
    def register(name: str, agent_class):
        AgentRegistry.agents[name] = agent_class


class SelfRegisteringAgent(Agent):
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    def __init_subclass__(cls, **kwargs):
        AgentRegistry.register(SelfRegisteringAgent._friendly_name(cls.__name__), cls)

    def name(self):
        return SelfRegisteringAgent._friendly_name(self.__class__.__name__)

    @staticmethod
    def _friendly_name(class_name: str):
        return class_name.replace("Agent", "").lower()


__all__ = ["RandomAgent", "SimpleAgent", "Agent", "FlatDQNAgent", "Pretrained01FlatDQNAgent"]

from agents.random import RandomAgent  # noqa: E402
from agents.simple import SimpleAgent  # noqa: E402
from agents.flat_dqn import FlatDQNAgent, Pretrained01FlatDQNAgent  # noqa: E402
