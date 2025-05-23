import copy
import math
import random
from typing import Optional

import numpy as np
from quoridor import Action, Player, Quoridor, construct_game_from_observation

from agents.core import Agent


class Node:
    def __init__(self, game: Quoridor, parent: Optional["Node"] = None, action_taken: Optional[Action] = None, depth=0):
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0

        # Just for debugging, remove
        self.depth = depth
        self.winner = ""
        # TODO maybe better to do it lazily
        self.valid_actions = list(game.get_valid_actions())
        random.shuffle(self.valid_actions)
        # print("Created a node, valid actions:", self.valid_actions)
        # TOOD arg
        self.ucb_c = 1.4

    def should_expand(self):
        return len(self.valid_actions) > 0 or len(self.children) == 0

    def expand(self):
        action = self.valid_actions.pop()

        # TO DO, this may take too much memory or be too slow
        game = copy.deepcopy(self.game)
        game.step(action)
        # dow e need to change the perspective?
        child = Node(game, self, action, self.depth + 1)
        self.children.append(child)
        return child

    def select(self) -> "Node":
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        assert best_child
        return best_child

    def get_ucb(self, child):
        # value_sum is in -1 or 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        # Then we invert it because we're switching players
        q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.ucb_c * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)

    # for debug
    def print(self):
        if self.action_taken:
            q_value = 1 - ((self.value_sum / self.visit_count) + 1) / 2

            print(
                f"{' ' * self.depth * 4}{self.game.action_encoder.action_to_index(self.action_taken)}: ({self.parent.get_ucb(self):.2f}) Q:{q_value} {self.value_sum}/{self.visit_count} {self.winner}"
            )
        else:
            print(f"{self.value_sum}/{self.visit_count}")
        for ch in self.children:
            ch.print()


class MCTS:
    def __init__(self, num_searches: int, game: Quoridor):
        self.num_searches = num_searches
        self.game = game

    def search(self, game: Quoridor):
        root = Node(game)
        player = game.current_player
        print("====== SEARCH ========")
        print(game.board)
        print(f"Player: {player}")

        for _ in range(self.num_searches):
            node = root

            while not node.should_expand():
                node = node.select()

            # TODO, ok? also, should this check be in the node?
            if node.game.check_win(player):
                terminal = True
                value = 1
                node.winner = "Me"
            elif node.game.check_win(1 - player):
                terminal = True
                value = -1
                node.winner = "Opp"
            else:
                terminal = False

            if not terminal:
                node = node.expand()

                # node.game.get_valid_actions()
                # TODO simulation
                value = 0

            node.backpropagate(value)

        root.print()

        # debug
        vc = -100000
        ch = None
        for c in root.children:
            if c.visit_count > vc:
                vc = c.visit_count
                ch = c

        return ch


class MCTSAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__()

        self.action_space = kwargs["action_space"]

    def start_game(self, game, player_id):
        self.player_id = player_id

    def get_action(self, observation):
        action_mask = observation["action_mask"]
        observation = observation["observation"]

        game, player, opponent = construct_game_from_observation(observation, self.player_id)
        mcts = MCTS(100, game)
        best = mcts.search(game)
        return game.action_encoder.action_to_index(best.action_taken)
        # return self.action_space.sample(action_mask)
