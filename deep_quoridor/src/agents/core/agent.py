class Agent:
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    @staticmethod
    def _friendly_name(class_name: str):
        return class_name.replace("Agent", "").lower()

    def name(self) -> str:
        return Agent._friendly_name(self.__class__.__name__)

    def is_trainable(self) -> bool:
        """Returns True if the agent is a learning agent, False otherwise."""
        return False

    def start_game(self, game, player_id):
        """This method is called when a new game starts.
        It allows the agent to reset its state if needed.
        """
        pass

    def end_game(self, game):
        """This method is called when a game ends.
        It allows the agent to clean up its state if needed.
        """
        pass

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
