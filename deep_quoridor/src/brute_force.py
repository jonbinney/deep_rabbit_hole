import argparse
import types

import numpy as np
import qgrid
from arena_utils import GameResult, MoveInfo
from numba import njit, prange
from plugins import ArenaYAMLRecorder
from quoridor import Board, Player, Quoridor, array_to_action

TIE_REWARD = 0
WIN_REWARD = 1
LARGE_REWARD = 2  # Used as "infinity"; must be bigger than winning reward

BOARD_SIZE = 0
WALL_SIZE = 0
WALL_BITS = 0
MAX_WALLS = 0
NUM_UNIQUE_WALLS = 0


@njit
def game_to_compact_id(board_size, grid, player_positions, walls_remaining):
    global BOARD_SIZE, WALL_SIZE, WALL_BITS
    row = 0
    col = 0
    compact_id = 0
    num_walls_found = 0
    while True:  # Vertical walls
        grid_i = row * 2 + 3
        grid_j = col * 2 + 3

        if grid[grid_i, grid_j] == qgrid.CELL_WALL:
            wall_id = grid_i * WALL_SIZE + grid_j
            compact_id += num_walls_found
            col += 2
        else:
            col += 1

        if col >= WALL_SIZE:
            col = 0
            row += 1

        if row >= WALL_SIZE:
            break


@njit(cache=False)
def minimax(
    action,
    grid,
    player_positions,
    walls_remaining,
    goal_rows,
    current_player,
    agent_player,
    depth,
    max_depth,
):
    opponent = 1 - current_player
    opponent_old_position = np.array([player_positions[opponent, 0], player_positions[opponent, 1]])

    # Apply the action the opponent just took
    qgrid.apply_action(grid, player_positions, walls_remaining, opponent, action)

    # Did we win?
    if qgrid.check_win(player_positions, goal_rows, current_player):
        best_value = WIN_REWARD if current_player == agent_player else -WIN_REWARD
        best_action_sequence = np.empty((0, 3), dtype=np.int8)

    # Did the opponent win?
    elif qgrid.check_win(player_positions, goal_rows, opponent):
        best_value = -WIN_REWARD if current_player == agent_player else WIN_REWARD
        best_action_sequence = np.empty((0, 3), dtype=np.int8)

    # Have we reached the maximum search depth?
    elif depth == max_depth:
        best_value = TIE_REWARD
        best_action_sequence = np.empty((0, 3), dtype=np.int8)

    # Try actions from this state.
    else:
        move_actions = qgrid.get_valid_move_actions(grid, player_positions, current_player)
        wall_actions = qgrid.get_valid_wall_actions(grid, player_positions, walls_remaining, goal_rows, current_player)
        next_actions = np.vstack((move_actions, wall_actions))
        assert len(next_actions) > 0, "No valid actions found"

        # Determine if maximizing or minimizing
        is_maximizing = current_player == agent_player
        best_value = -np.inf if is_maximizing else np.inf
        best_action_sequence = np.full((max_depth - depth, 3), -1, dtype=np.int8)

        # Evaluate all actions
        for i in range(len(next_actions)):
            next_action = next_actions[i]

            # Recursively evaluate position
            value, future_best_action_sequence = minimax(
                next_action,
                grid,
                player_positions,
                walls_remaining,
                goal_rows,
                1 - current_player,
                agent_player,
                depth + 1,
                max_depth,
            )

            # Update best action
            if is_maximizing:
                if value > best_value:
                    best_value = value
                    best_action_sequence[0:1, :] = next_action
                    best_action_sequence[1 : 1 + len(future_best_action_sequence), :] = future_best_action_sequence

                    if best_value == WIN_REWARD:
                        break
            else:  # Minimizing
                if value < best_value:
                    best_value = value
                    best_action_sequence[0:1, :] = next_action
                    best_action_sequence[1 : 1 + len(future_best_action_sequence), :] = future_best_action_sequence

                    if best_value == -WIN_REWARD:
                        break

    # Undo action
    qgrid.undo_action(grid, player_positions, walls_remaining, opponent, action, opponent_old_position)

    return best_value, best_action_sequence


@njit(cache=False, parallel=True)
def evaluate_actions(
    grid,
    player_positions,
    walls_remaining,
    goal_rows,
    current_player,
    max_depth,
):
    """
    Evaluate all actions for the current player using the minimax algorithm.
    """
    # Sample actions to evaluate
    move_actions = qgrid.get_valid_move_actions(grid, player_positions, current_player)
    wall_actions = qgrid.get_valid_wall_actions(grid, player_positions, walls_remaining, goal_rows, current_player)
    actions = np.vstack((move_actions, wall_actions))
    assert len(actions) > 0, "No valid actions found"

    # Evaluate all actions
    values = np.zeros(len(actions), dtype=np.int8)
    best_action_sequences = np.full((len(actions), max_depth, 3), -1, np.int8)
    for i in prange(len(actions)):
        best_action_sequences[i : i + 1, 0:1, :] = actions[i]

        print(f"Evaluating action {i}/{len(actions)}")
        # Since we run this loop in parallel, we need to copy the game state arrays for each minimax call
        values[i], future_best_action_sequence = minimax(
            actions[i],
            grid.copy(),
            player_positions.copy(),
            walls_remaining.copy(),
            goal_rows,
            1 - current_player,
            current_player,  # Assume we are choosing an action for the current player.
            1,
            max_depth,
        )
        best_action_sequences[i, 1:, :] = future_best_action_sequence

    return actions, values, best_action_sequences


def build_optimal_action_tree(board_size: int, max_walls: int, max_depth: int):
    game = Quoridor(Board(board_size, max_walls))

    # Convert the game state to arrays that can be used by Numba
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

    # Use Numba-optimized minimax to evaluate possible actions
    actions, values, best_action_sequences = evaluate_actions(
        grid,
        player_positions,
        walls_remaining,
        goal_rows,
        current_player,
        max_depth,
    )

    def a2s(action):
        return f"{action[0]}{action[1]}{action[2]}"

    recorder = ArenaYAMLRecorder("replay.yaml")
    for action_i in range(len(actions)):
        print(f"{action_i}: (v={values[action_i]}) - " + ",".join([a2s(m) for m in best_action_sequences[action_i]]))
        recorder.start_game(None, None, None)

        move_infos = []
        for m in best_action_sequences[action_i]:
            if (m == -1).all():
                break

            action_object = array_to_action(m)
            action_index = game.action_encoder.action_to_index(action_object)
            move_infos.append(MoveInfo(f"p{(len(move_infos) % 1) + 1}", action_index, 0.0))
            recorder.after_action(None, None, None, int(action_index))

        if values[action_i] == WIN_REWARD:
            winner = "p1"
        elif values[action_i] == -WIN_REWARD:
            winner = "p2"
        elif values[action_i] == 0.0:
            winner = "tie"
        else:
            winner = "bad value"

        recorder.end_game(
            None,
            GameResult(
                "p1",
                "p2",
                winner,
                len(move_infos),
                0,
                f"{action_i}",
                move_infos,
            ),
        )

    game_mock = types.SimpleNamespace()
    game_mock.board_size = board_size
    game_mock.max_walls = board_size
    game_mock.step_rewards = False
    recorder.end_arena(game_mock, None)


def main():
    parser = argparse.ArgumentParser(description="Build a tree of optimal moves")
    parser.add_argument("board_size", type=int)
    parser.add_argument("max_walls", type=int)
    parser.add_argument("max_depth", type=int)
    args = parser.parse_args()
    print(f"Board size:{args.board_size} Max walls:{args.max_walls} Max depth:{args.max_depth}")
    build_optimal_action_tree(args.board_size, args.max_walls, args.max_depth)


if __name__ == "__main__":
    main()
