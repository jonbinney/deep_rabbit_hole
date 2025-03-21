class Agent:
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    def name(self) -> str:
        raise NotImplementedError("You must implement the name method")

    def get_action(self, game) -> int:
        raise NotImplementedError("You must implement the get_action method")
