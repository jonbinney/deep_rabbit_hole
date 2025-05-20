from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import qgrid
from numba import njit
from quoridor import Action, ActionEncoder, MoveAction, Player, Quoridor, WallAction, construct_game_from_observation
from utils import SubargsBase

from agents.core import Agent

# We use a large reward to encourage the agent to win the game, but we can't use infinity
# because then multiplying by a discount factor won't decrease the reward.
WINNING_REWARD = 1e6

# Constants for action type
ACTION_MOVE = 0
ACTION_WALL_VERTICAL = 1
ACTION_WALL_HORIZONTAL = 2


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


@njit
def get_move_actions(grid, player_positions, current_player, board_size):
    """
    Get all valid move actions (Numba-optimized).
    Returns an array of actions where each row is [row, col, action_type=0].
    """
    max_actions = board_size * board_size  # Maximum possible moves
    actions = np.zeros((max_actions, 3), dtype=np.int32)
    count = 0

    # Check all possible moves on the board
    for dest_row in range(board_size):
        for dest_col in range(board_size):
            if qgrid.is_move_valid(grid, player_positions, current_player, dest_row, dest_col):
                actions[count, 0] = dest_row
                actions[count, 1] = dest_col
                actions[count, 2] = ACTION_MOVE
                count += 1

    return actions[:count]  # Return only valid actions


@njit
def get_wall_actions(grid, player_positions, walls_remaining, goal_rows, current_player, board_size):
    """
    Get all valid wall actions (Numba-optimized).
    Returns an array of actions where each row is [row, col, action_type] with action_type 1 or 2.
    """
    wall_size = board_size - 1
    max_actions = wall_size * wall_size * 2  # Maximum possible wall placements
    actions = np.zeros((max_actions, 3), dtype=np.int32)
    count = 0

    # Check all possible wall placements
    for wall_row in range(wall_size):
        for wall_col in range(wall_size):
            # Check vertical walls
            if qgrid.is_wall_action_valid(
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                qgrid.WALL_ORIENTATION_VERTICAL,
            ):
                actions[count, 0] = wall_row
                actions[count, 1] = wall_col
                actions[count, 2] = ACTION_WALL_VERTICAL
                count += 1

            # Check horizontal walls
            if qgrid.is_wall_action_valid(
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                current_player,
                wall_row,
                wall_col,
                qgrid.WALL_ORIENTATION_HORIZONTAL,
            ):
                actions[count, 0] = wall_row
                actions[count, 1] = wall_col
                actions[count, 2] = ACTION_WALL_HORIZONTAL
                count += 1

    return actions[:count]  # Return only valid actions


@njit
def gaussian_wall_weights(wall_actions, p1_pos, p2_pos, sigma):
    """
    Calculate Gaussian weights for wall actions based on distance to players (Numba-optimized).
    """
    num_actions = len(wall_actions)
    weights = np.zeros(num_actions, dtype=np.float32)

    for i in range(num_actions):
        wall_row, wall_col = wall_actions[i, 0], wall_actions[i, 1]
        wall_pos = np.array([wall_row + 0.5, wall_col + 0.5], dtype=np.float32)

        # Calculate distance to both players
        d1 = (wall_pos[0] - p1_pos[0]) ** 2 + (wall_pos[1] - p1_pos[1]) ** 2
        d2 = (wall_pos[0] - p2_pos[0]) ** 2 + (wall_pos[1] - p2_pos[1]) ** 2
        min_dist = min(d1, d2)

        # Calculate Gaussian weight
        weights[i] = np.exp(-0.5 * min_dist / (sigma**2))

    # Normalize weights
    weights_sum = np.sum(weights)
    if weights_sum > 0:
        weights = weights / weights_sum
    else:
        weights.fill(1.0 / num_actions)

    return weights


@njit
def sample_actions(
    grid, player_positions, walls_remaining, goal_rows, current_player, board_size, all_move_actions, all_wall_actions, branching_factor, wall_sigma=0.5
):
    """
    Sample actions for the minimax search (Numba-optimized).
    Returns an array of actions where each row is [row, col, action_type].
    """
    # First get all valid move actions
    move_actions = all_move_actions[qgrid.]

    # Check if we already have enough move actions
    if len(move_actions) >= branching_factor:
        # Randomly select branching_factor moves
        indices = np.arange(len(move_actions))
        np.random.shuffle(indices)
        return move_actions[indices[:branching_factor]]

    # Calculate how many wall actions we need
    num_wall_actions_needed = branching_factor - len(move_actions)

    # Get all valid wall actions
    wall_actions = get_wall_actions(grid, player_positions, walls_remaining, goal_rows, current_player, board_size)

    # Combine the actions
    if len(wall_actions) <= num_wall_actions_needed:
        # Use all wall actions
        combined_actions = np.zeros((len(move_actions) + len(wall_actions), 3), dtype=np.int32)
        combined_actions[: len(move_actions)] = move_actions
        combined_actions[len(move_actions) :] = wall_actions
        return combined_actions
    else:
        # Sample wall actions based on distance to players
        if wall_sigma > 0:
            weights = gaussian_wall_weights(wall_actions, player_positions[0], player_positions[1], wall_sigma)

            # Sample wall actions using weights
            indices = np.zeros(num_wall_actions_needed, dtype=np.int32)
            cumulative = np.cumsum(weights)

            for i in range(num_wall_actions_needed):
                r = np.random.random()
                for j in range(len(cumulative)):
                    if r <= cumulative[j]:
                        indices[i] = j
                        break
        else:
            # Sample wall actions uniformly
            indices = np.zeros(num_wall_actions_needed, dtype=np.int32)
            temp_indices = np.arange(len(wall_actions))
            np.random.shuffle(temp_indices)
            for i in range(min(num_wall_actions_needed, len(temp_indices))):
                indices[i] = temp_indices[i]

        # Combine move actions with sampled wall actions
        sampled_wall_actions = wall_actions[indices]
        combined_actions = np.zeros((len(move_actions) + len(sampled_wall_actions), 3), dtype=np.int32)
        combined_actions[: len(move_actions)] = move_actions
        combined_actions[len(move_actions) :] = sampled_wall_actions

        return combined_actions


@njit
def compute_heuristic_for_game_state(grid, player_positions, walls_remaining, goal_rows, agent_player):
    """
    Evaluate a board position (Numba-optimized).
    """
    # Get distances to goals
    opponent = 1 - agent_player
    agent_distance = qgrid.distance_to_row(
        grid, player_positions[agent_player, 0], player_positions[agent_player, 1], goal_rows[agent_player]
    )
    opponent_distance = qgrid.distance_to_row(
        grid, player_positions[opponent, 0], player_positions[opponent, 1], goal_rows[opponent]
    )

    assert agent_distance != -1 and opponent_distance != -1

    # Compute heuristic value based on distances and walls
    distance_reward = opponent_distance - agent_distance
    wall_reward = (walls_remaining[agent_player] - walls_remaining[opponent]) / 100.0

    return distance_reward + wall_reward


@njit
def minimax(
    action,
    grid,
    player_positions,
    walls_remaining,
    goal_rows,
    current_player,
    agent_player,
    depth,
    branching_factor,
    wall_sigma,
    discount_factor,
):
    # Apply action
    old_position = np.array([player_positions[current_player, 0], player_positions[current_player, 1]])
    qgrid.apply_action(grid, player_positions, walls_remaining, current_player, action)
    opponent = 1 - current_player

    # Did we win?
    if qgrid.check_win(player_positions, goal_rows, current_player):
        best_value = WINNING_REWARD if current_player == agent_player else -WINNING_REWARD

    # Did the opponent win?
    elif qgrid.check_win(player_positions, goal_rows, opponent):
        best_value = -WINNING_REWARD if current_player == agent_player else WINNING_REWARD

    # Have we reached the maximum depth?
    elif depth == 0:
        best_value = compute_heuristic_for_game_state(grid, player_positions, walls_remaining, goal_rows, agent_player)

    # Try actions from this state.
    else:
        # Sample actions to evaluate
        next_actions = sample_actions(
            grid,
            player_positions,
            walls_remaining,
            goal_rows,
            current_player,
            len(goal_rows),
            branching_factor,
            wall_sigma,
        )
        assert len(next_actions) > 0, "No valid actions found"

        # Determine if maximizing or minimizing
        is_maximizing = current_player == agent_player
        best_value = -np.float32(WINNING_REWARD * 2) if is_maximizing else np.float32(WINNING_REWARD * 2)

        # Evaluate all actions
        for i in range(len(next_actions)):
            next_action = next_actions[i]

            # Recursively evaluate position
            value = minimax(
                next_action,
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                1 - current_player,
                agent_player,
                depth - 1,
                branching_factor,
                wall_sigma,
                discount_factor,
            )

            # Apply discount factor
            value *= discount_factor

            # Update best value
            if is_maximizing:
                best_value = max(best_value, value)
            else:
                best_value = min(best_value, value)

    # Undo action
    qgrid.undo_action(grid, player_positions, walls_remaining, current_player, action, old_position)

    return best_value


@njit
def choose_action_numba(
    grid,
    player_positions,
    walls_remaining,
    goal_rows,
    current_player,
    max_depth,
    branching_factor,
    wall_sigma,
    discount_factor,
):
    """
    Choose the best action using minimax (Numba-optimized).
    Returns the action array [row, col, action_type] and its value.
    """
    # Sample actions to evaluate
    actions = sample_actions(
        grid, player_positions, walls_remaining, goal_rows, current_player, len(goal_rows), branching_factor, wall_sigma
    )
    assert len(actions) > 0, "No valid actions found"

    # Evaluate all actions
    values = np.zeros(len(actions), dtype=np.float32)
    for i in range(len(actions)):
        values[i] = minimax(
            actions[i],
            grid,
            player_positions,
            walls_remaining,
            goal_rows,
            1 - current_player,
            current_player,  # Assume we are choosing an action for the current player.
            max_depth - 1,
            branching_factor,
            wall_sigma,
            discount_factor,
        )

    # If multiple actions have the same value, choose randomly among them
    best_value = np.max(values)
    indices = np.flatnonzero(values == best_value)
    assert len(indices) > 0, "No best action found"
    if len(indices) == 1:
        best_action_i = indices[0]
    else:
        best_action_i = indices[np.random.randint(0, len(indices))]

    return actions[best_action_i]


def choose_action(
    game: Quoridor,
    max_depth: int,
    branching_factor: int,
    wall_sigma: float | None,
    discount_factor: float,
) -> tuple[Optional[Action], float]:
    """
    Minimax algorithm to choose the best action for the current player.
    This is a wrapper around the Numba-optimized implementation.
    """
    # Extract the game state as NumPy arrays
    grid = game.board._grid
    player_positions = np.zeros((2, 2), dtype=np.int32)
    player_positions[0] = game.board.get_player_position(Player.ONE)
    player_positions[1] = game.board.get_player_position(Player.TWO)

    walls_remaining = np.zeros(2, dtype=np.int32)
    walls_remaining[0] = game.board.get_walls_remaining(Player.ONE)
    walls_remaining[1] = game.board.get_walls_remaining(Player.TWO)

    goal_rows = np.zeros(2, dtype=np.int32)
    goal_rows[0] = game.get_goal_row(Player.ONE)
    goal_rows[1] = game.get_goal_row(Player.TWO)

    current_player = int(game.get_current_player())
    agent_player = current_player

    # Use Numba-optimized minimax to choose the best action
    sigma = wall_sigma if wall_sigma is not None else 0.5
    action = choose_action_numba(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
        agent_player,
        max_depth,
        branching_factor,
        sigma,
        discount_factor,
    )

    # Convert the action to a Quoridor Action object
    if action[0] == -1:  # No valid action found
        return None

    return action


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

        chosen_action = choose_action(
            game,
            self.params.max_depth,
            self.params.branching_factor,
            self.params.wall_sigma,
            self.params.discount_factor,
        )

        assert game.is_action_valid(chosen_action), "The chosen action is not valid."
        action_id = self.action_encoder.array_to_index(chosen_action)
        assert action_mask[action_id] == 1, "The action is not valid according to the action mask."

        return action_id
