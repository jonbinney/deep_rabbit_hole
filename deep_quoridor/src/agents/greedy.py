from queue import Queue

import numpy as np
from quoridor import Player

from agents.core import Agent


class GreedyAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

    def start_game(self, game, player_id):
        self.player_id = player_id
        print(player_id)

    def _valid_pawn_movements(self, game, observation):
        mask = observation["action_mask"]
        N = game.board_size
        movement_mask = mask[: N * N]
        valid_moves = [i for i, x in enumerate(movement_mask) if x == 1]
        print("Valid moves:", [game.action_index_to_params(i)[:2] for i in valid_moves])
        return [game.action_index_to_params(i)[:2] for i in valid_moves]

    def _next_moves(self, game, pos):
        moves = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row = pos[0] + dr
            col = pos[1] + dc
            if game.is_in_board(row, col) and not game.is_wall_between(pos[0], pos[1], row, col):
                moves.append((row, col))
        return moves

    def _shortest_path_from(self, game, pos, target):
        if pos[0] == target:
            return []

        distances = [[None] * game.board_size for _ in range(game.board_size)]

        q = Queue()
        q.put((0, pos[0], pos[1]))
        distances[pos[0]][pos[1]] = 0
        dest = None
        coming_from = {}

        while not q.empty() and dest is None:
            d, r, c = q.get()
            distances[r][c] = d

            for next_r, next_c in self._next_moves(game, (r, c)):
                if distances[next_r][next_c] is None:
                    distances[next_r][next_c] = d + 1
                    q.put((d + 1, next_r, next_c))
                    coming_from[(next_r, next_c)] = (r, c)
                    if next_r == target:
                        dest = (next_r, next_c)
        if dest is None:
            print(pos, target, distances)
        assert dest is not None
        path = [dest]
        while True:
            last_pos = path[-1]
            prev_pos = coming_from[last_pos]
            if prev_pos == pos:
                break

            path.append(prev_pos)

        print(path[::-1])
        return path[::-1]

    def _shortest_path(self, game, observation, target) -> list:
        first_moves = self._valid_pawn_movements(game, observation)
        shortest_path = None
        for move in first_moves:
            path = self._shortest_path_from(game, move, target)
            if shortest_path is None or len(path) < len(shortest_path):
                shortest_path = [move] + path

        assert shortest_path is not None, "No path found"
        return shortest_path

    def _get_opponent_position(self, game, observation):
        board = observation["observation"]["board"]
        coords = tuple(int(x[0]) for x in np.where(board == 2))
        assert len(coords) == 2, "Expected exactly one opponent position"
        return coords

    def get_action(self, game):
        observation, _, termination, truncation, _ = game.last()
        if termination or truncation:
            return None

        goal_row = game.board_size - 1 if self.player_id == "player_0" else 0
        opp_row = game.board_size - 1 - goal_row

        my_shortest_path = self._shortest_path(game, observation, goal_row)
        pos = self._get_opponent_position(game, observation)
        opp_shortest_path = self._shortest_path_from(game, pos, opp_row)
        print(f"opponent: {pos} my shortest path: {my_shortest_path}, opp: {opp_shortest_path}")

        block = False
        if len(opp_shortest_path) < len(my_shortest_path):
            if len(opp_shortest_path) < 5:
                block = True

        if block:
            mask = observation["action_mask"]
            print("I'm loosing, block!")
            action = None
            for p0, p1 in zip(opp_shortest_path, opp_shortest_path[1:]):
                print(p0, p1)
                if p0[0] == p1[0]:
                    print("HHHH==========")
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

                print("QQQQ")
                for r in rows:
                    for c in cols:
                        print(f"Trying with {r}, {c}, {orientation}")
                        if r < 0 or c < 0 or r >= game.wall_size or c >= game.wall_size:
                            continue
                        action = game.action_params_to_index(r, c, orientation)
                        print(action)
                        print(mask[action], mask)
                        if mask[action] == 0:
                            print("invalid", action)
                            action = None
                            continue
                        print(f"Returnign wall {action}")
                        return action
                        # self.walls[row, col, orientation] = 1
                        # Check if it's allowed

                # print("Would try blockig", rows, cols)
                if action is not None:  # May not have possibilities to put a wall
                    return action

            # TODO FIND THE BEST
            if action is not None:  # May not have possibilities to put a wall
                return action

        return game.action_params_to_index(my_shortest_path[0][0], my_shortest_path[0][1], 0)
