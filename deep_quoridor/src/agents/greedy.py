from queue import Queue
from typing import TypeAlias

import numpy as np

from agents.core import Agent

Position: TypeAlias = tuple[int, int]


class GreedyAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

    def start_game(self, game, player_id):
        self.player_id = player_id

    def _valid_pawn_movements(self, game, action_mask: np.ndarray) -> list[Position]:
        """Returns the coordinates of the possible movements based on the action mask.
        This is useful for the first move, since there could be complicated jumps.  After the
        first move, you can't easily use this because you'd need to generate a new action_mask
        """
        movement_mask = action_mask[: game.board_size**2]
        valid_moves = np.argwhere(movement_mask == 1).reshape(-1)
        return [game.action_index_to_params(i)[:2] for i in valid_moves]

    def _next_moves(self, game, pos: Position) -> list[Position]:
        """Given a position in the board, it returns the possible next moves.  Notice that it ignores
        the other pawn, because this function is meant to be used for moves other than the first one,
        so the pawn may not be there anymore.
        """
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row = pos[0] + dr
            col = pos[1] + dc
            if game.is_in_board(row, col) and not game.is_wall_between(pos[0], pos[1], row, col):
                moves.append((row, col))
        return moves

    def _shortest_path_from(self, game, pos: Position, target_row: int) -> list[Position]:
        """
        Given a position in the board, it returns the shortest path to get to the target_row.
        """
        if pos[0] == target_row:
            return []

        # How many steps are needed to get to that position, starting from pos (None means unreachable)
        distances = [[None] * game.board_size for _ in range(game.board_size)]

        # Queue for the BFS, containing the neighbords to process
        q = Queue()
        q.put((0, pos[0], pos[1]))

        # Coordinates of the destination when the target row was reached, or None if it wasn't
        dest = None

        # Map from position to previous position, to trace back the path
        coming_from = {}

        while not q.empty() and dest is None:
            d, r, c = q.get()
            distances[r][c] = d

            for next_r, next_c in self._next_moves(game, (r, c)):
                if distances[next_r][next_c] is None:
                    distances[next_r][next_c] = d + 1
                    q.put((d + 1, next_r, next_c))
                    coming_from[(next_r, next_c)] = (r, c)
                    if next_r == target_row:
                        dest = (next_r, next_c)

        assert dest is not None, f"Can't find a path from {pos} to {target_row} row"

        # Trace back the path
        path = [dest]
        while True:
            last_pos = path[-1]
            prev_pos = coming_from[last_pos]
            if prev_pos == pos:
                break

            path.append(prev_pos)

        return path[::-1]

    def _shortest_path_from_me(self, game, observation: dict, target_row: int) -> list[Position]:
        """Return the shortest path from wherever the agent currently is."""
        # The first move is based on the action mask, that already calculated what positions are valid
        first_moves = self._valid_pawn_movements(game, observation["action_mask"])
        shortest_path = None

        # Find the shortest path for each of the first moves and keep the shortest one.
        for move in first_moves:
            path = self._shortest_path_from(game, move, target_row)
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = [move] + path

        assert shortest_path is not None, "No path found"
        return shortest_path

    def _get_opponent_position(self, game, board: np.ndarray) -> Position:
        coords = np.argwhere(board == 2)
        assert len(coords) == 1, "Expected exactly one opponent position"
        return (int(coords[0][0]), int(coords[0][1]))

    def _get_block_action(self, game, opponent_shortest_path: list[Position], action_mask: np.ndarray):
        for p0, p1 in zip(opponent_shortest_path, opponent_shortest_path[1:]):
            # For this movement of the opponent, get the possible wall placements
            if p0[0] == p1[0]:
                # Horizontal movement
                rows = [p0[0], p0[0] - 1]
                cols = [min(p0[1], p1[1])]
                orientation = 1
            elif p0[1] == p1[1]:
                # Vertical movement
                rows = [min(p0[0], p1[0])]
                cols = [p0[1], p0[1] - 1]
                orientation = 2
            else:
                assert "Expected horizontal or vertical movement"

            # Check if any of the wall placements are valid.
            # TO DO: instead of just returning the first, we can check which one is better
            # by looking into which makes the difference of the distances more favorable
            for r in rows:
                for c in cols:
                    if r < 0 or c < 0 or r >= game.wall_size or c >= game.wall_size:
                        continue

                    action = game.action_params_to_index(r, c, orientation)
                    if action_mask[action] == 1:
                        return action
        # No actions found
        return None

    def _log_action(
        self, game, observation: dict, my_shortest_path: list[Position], opponent_shortest_path: list[Position]
    ):
        if not self.log.is_enabled():
            return

        self.log.clear()

        # Log the possible next movements
        movement_mask = observation["action_mask"][: game.board_size**2]
        for action in np.argwhere(movement_mask == 1).reshape(-1):
            self.log.action_text(int(action), "")

        my_coords = np.argwhere(observation["observation"]["board"] == 1)
        path = my_shortest_path[:]
        path.insert(0, (int(my_coords[0][0]), int(my_coords[0][1])))
        self.log.path(path)

        opp_coords = np.argwhere(observation["observation"]["board"] == 2)
        path = opponent_shortest_path[:]
        path.insert(0, (int(opp_coords[0][0]), int(opp_coords[0][1])))
        self.log.path(path)

        # opp_coords = np.argwhere(observation["observation"]["board"] == 2)
        # qq = opponent_shortest_path[:]
        # qq.insert(0, opponent_pos)  # (int(opp_coords[0][0]), int(opp_coords[0][1])))
        # self.log.path(qq)

    def get_action(self, game):
        observation, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        goal_row = game.board_size - 1 if self.player_id == "player_0" else 0
        opponent_row = game.board_size - 1 - goal_row

        my_shortest_path = self._shortest_path_from_me(game, observation, goal_row)
        opponent_pos = self._get_opponent_position(game, observation["observation"]["board"])
        opponent_shortest_path = self._shortest_path_from(game, opponent_pos, opponent_row)

        self._log_action(game, observation, my_shortest_path, opponent_shortest_path)

        # TODO: use a more elaborate logic to decide whether to block, could be probabilistic
        block = False
        if len(opponent_shortest_path) < len(my_shortest_path):
            if len(opponent_shortest_path) < 5:
                block = True
        if block:
            action = self._get_block_action(game, opponent_shortest_path, observation["action_mask"])
            if action is not None:
                return action

        return game.action_params_to_index(my_shortest_path[0][0], my_shortest_path[0][1], 0)
