from agents import Agent


class AgentRegistry:
    agents = {}

    @staticmethod
    def create(friendly_name: str) -> Agent:
        return AgentRegistry.agents[friendly_name]()

    @staticmethod
    def names():
        return list(AgentRegistry.agents.keys())

    @staticmethod
    def register(name: str, agent_class):
        AgentRegistry.agents[name] = agent_class


class SelfRegisteringAgent:
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
