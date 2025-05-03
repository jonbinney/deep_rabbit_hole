import copy
import random

from quoridor import ActionEncoder, Player, Quoridor, construct_game_from_observation

from agents.core import Agent


def sample_valid_actions(game: Quoridor, n: int):
    """
    Choose a random sample of valid actions for the current player.

    Starts by including all valid move actions, then adds a random sample of valid wall actions. May return less than
    n actions if there are not enough valid actions.
    """
    actions = game.get_valid_move_actions()
    if len(actions) > n:
        actions = random.sample(actions, n)

    if len(actions) < n:
        num_wall_actions = n - len(actions)
        wall_actions = game.get_valid_wall_actions()
        if len(wall_actions) > num_wall_actions:
            wall_actions = random.sample(wall_actions, num_wall_actions)

        actions += wall_actions

    return actions


def choose_action(game: Quoridor, player: Player, opponent: Player, max_depth: int, branching_factor: int) -> float:
    """
    Minimax algorithm to choose the best action for the current player.

    The other agent tries to minimize the reward of the current player.
    """
    if game.check_win(player):
        return None, float("inf")
    elif game.check_win(opponent):
        return None, -float("inf")
    elif max_depth == 0:
        return None, game.player_distance_to_target(opponent) - game.player_distance_to_target(player)

    actions = sample_valid_actions(game, branching_factor)
    assert len(actions) > 0, "There are no valid actions for the current player."

    chosen_action = None
    chosen_value = -float("inf") if game.get_current_player() == player else float("inf")
    for action in actions:
        game_after_action = copy.deepcopy(game)
        game_after_action.step(action)
        _, value = choose_action(game_after_action, player, opponent, max_depth - 1, branching_factor)
        if game.get_current_player() == player:
            if value >= chosen_value:
                chosen_value = value
                chosen_action = action
        else:
            if value <= chosen_value:
                chosen_value = value
                chosen_action = action

    return chosen_action, chosen_value


class SimpleAgent(Agent):
    def __init__(self, max_depth=3, branching_factor=8, **kwargs):
        super().__init__()
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.board_size = kwargs["board_size"]
        self.action_encoder = ActionEncoder(self.board_size)

    def start_game(self, game, player_id):
        self.player_id = player_id

    def get_action(self, observation, action_mask):
        game, player, opponent = construct_game_from_observation(observation, self.player_id)

        chosen_action, chosen_value = choose_action(game, player, opponent, self.max_depth, self.branching_factor)

        assert game.is_action_valid(chosen_action), "The chosen action is not valid."
        action_id = self.action_encoder.action_to_index(chosen_action)
        assert action_mask[action_id] == 1, "The action is not valid according to the action mask."

        return action_id
