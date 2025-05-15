import random
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
from quoridor import Action, ActionEncoder, Player, Quoridor, WallAction, construct_game_from_observation
from utils import SubargsBase

from agents.core import Agent

# We use a large reward to encourage the agent to win the game, but we can't use infinity
# because then multiplying by a discount factor won't decrease the reward.
WINNING_REWARD = 1e6


@dataclass
class SimpleParams(SubargsBase):
    nick: Optional[str] = None

    # How many moves to look ahead in the minimax algorithm.
    max_depth: int = 2

    # How many actions to consider at each minimax stage.
    branching_factor: int = 20

    # We use a Guassian distribution centered around each player's position to sample wall actions.
    # Sigma is the standard deviation of this distribution; a smaller sigma means the agent is more
    # likely to place walls near themselves or their opponent, rather than somewhere else on the board.
    wall_sigma: float = 0.5

    # Future rewards are worth less. This makes the agent try to win quickly. Without this,
    # once the agent knows it can win, it may oscillate between two moves instead of just winning.
    discount_factor: float = 0.99


def compute_wall_weights(
    wall_actions: list[WallAction], position_1: tuple[int, int], position_2: tuple[int, int], sigma
) -> float:
    walls_array = np.array([wall.position for wall in wall_actions], dtype=np.float32)
    walls_array += (0.5, 0.5)  # Adjust wall positions to reflect where their center is in player coordinates.
    squared_distances_to_position_1 = ((walls_array - position_1) ** 2).sum(axis=1)
    squared_distances_to_position_2 = ((walls_array - position_2) ** 2).sum(axis=1)
    min_squared_distances = np.minimum(squared_distances_to_position_1, squared_distances_to_position_2)
    weights = np.exp(-0.5 * min_squared_distances / sigma**2)
    return weights


def sample_actions(game: Quoridor, n: int, wall_sigma: float | None = None) -> list[Action]:
    """
    Choose a sample of valid actions for the current player.

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
            if wall_sigma is None:
                # Use a uniform distribution to sample wall actions.
                wall_actions = random.sample(wall_actions, num_wall_actions)
            else:
                # Weight walls based on their distance to the players.
                position_1 = game.board.get_player_position(Player.ONE)
                position_2 = game.board.get_player_position(Player.TWO)
                wall_weights = compute_wall_weights(wall_actions, position_1, position_2, wall_sigma)
                wall_actions = np.random.choice(
                    wall_actions,
                    size=num_wall_actions,
                    replace=False,
                    p=wall_weights / np.sum(wall_weights),
                )

        actions.extend(wall_actions)

    return actions


def choose_action(
    game: Quoridor,
    player: Player,
    opponent: Player,
    max_depth: int,
    branching_factor: int,
    wall_sigma: float | None,
    discount_factor: float,
) -> tuple[Optional[Action], float]:
    """
    Minimax algorithm to choose the best action for the current player.

    The other agent tries to minimize the reward of the current player.
    """
    if game.check_win(player):
        return None, WINNING_REWARD
    elif game.check_win(opponent):
        return None, -WINNING_REWARD
    elif max_depth == 0:
        distance_reward = game.player_distance_to_target(opponent) - game.player_distance_to_target(player)
        wall_reward = game.board.get_walls_remaining(player) - game.board.get_walls_remaining(opponent)
        return None, distance_reward + wall_reward / 100

    actions = sample_actions(game, branching_factor, wall_sigma)
    assert len(actions) > 0, "There are no valid actions for the current player."

    values = []
    current_player_before_action = game.get_current_player()
    position_before_action = game.board.get_player_position(current_player_before_action)
    for action in actions:
        # Apply the action to the game.
        game.step(action)

        # Recursively call this function to choose the next player's action.
        _, value = choose_action(game, player, opponent, max_depth - 1, branching_factor, wall_sigma, discount_factor)
        value = discount_factor * value
        values.append(value)

        # Undo the action to put the game back to its original state.
        if isinstance(action, WallAction):
            game.board.remove_wall(current_player_before_action, action.position, action.orientation)
        else:
            game.board.move_player(current_player_before_action, position_before_action)
        game.set_current_player(current_player_before_action)

    # "player" tries to maximize the reward, while "opponent" tries to minimize it.
    values = np.array(values)
    if current_player_before_action == player:
        best_value = values.max()
    else:
        best_value = values.min()

    best_action_indices = np.flatnonzero(values == best_value)
    # There may be multiple actions with the same value, so we randomly choose one of them.
    # chosen_action_index = np.random.choice(best_action_indices)
    chosen_action_index = best_action_indices[-1]

    return actions[chosen_action_index], values[chosen_action_index]


class SimpleAgent(Agent):
    def __init__(self, params=SimpleParams(), **kwargs):
        super().__init__()
        self.params = params
        self.board_size = kwargs["board_size"]
        self.action_encoder = ActionEncoder(self.board_size)

    @classmethod
    def params_class(cls):
        return SimpleParams

    def name(self):
        if self.params.nick:
            return self.params.nick
        param_strings = []
        for f in fields(self.params):
            value = getattr(self.params, f.name)
            if f.name != "nick" and value != f.default:
                param_strings.append(f"{f.name}={getattr(self.params, f.name)}")
        return "simple " + ",".join(param_strings)

    def start_game(self, game, player_id):
        self.player_id = player_id

    def get_action(self, observation):
        action_mask = observation["action_mask"]
        observation = observation["observation"]

        game, player, opponent = construct_game_from_observation(observation, self.player_id)

        chosen_action, chosen_value = choose_action(
            game,
            player,
            opponent,
            self.params.max_depth,
            self.params.branching_factor,
            self.params.wall_sigma,
            self.params.discount_factor,
        )

        assert game.is_action_valid(chosen_action), "The chosen action is not valid."
        action_id = self.action_encoder.action_to_index(chosen_action)
        assert action_mask[action_id] == 1, "The action is not valid according to the action mask."

        return action_id
