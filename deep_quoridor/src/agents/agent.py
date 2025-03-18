class Agent:
    """
    Base class for all agents.
    Given a game state, the agent should return an action.
    """

    def get_action(self, game):
        raise NotImplementedError("You must implement the get_action method")
