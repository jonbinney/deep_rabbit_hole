import copy
import math
import random
from dataclasses import dataclass
from typing import Optional

from quoridor import Action, Player, Quoridor, construct_game_from_observation
from utils.subargs import SubargsBase

from agents.core import Agent


@dataclass
class MCTSParams(SubargsBase):
    nick: Optional[str] = None
    # Number of searches
    n: int = 1000
    # A higher number favors exploration over exploitation
    c: float = 1.4
    # Maximum number of steps during random simulation before declaring a tie
    max_steps: int = 1000

    # How to evaluate the value after the node expansion.
    # - rand_sim: random simulation of a game until there's a winner or max_steps are reached
    # - dist: based on the distance of the pawns to the target and the number of remaining walls
    eval: str = "dist"


class Node:
    def __init__(
        self, game: Quoridor, parent: Optional["Node"] = None, action_taken: Optional[Action] = None, ucb_c: float = 1.0
    ):
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        self.valid_actions = list(game.get_valid_actions())
        random.shuffle(self.valid_actions)

        self.ucb_c = ucb_c

    def should_expand(self):
        return len(self.valid_actions) > 0 or len(self.children) == 0

    def expand(self):
        """
        Create a child of the current node.
        """
        action = self.valid_actions.pop()

        game = copy.deepcopy(self.game)
        game.step(action)

        child = Node(game, parent=self, action_taken=action, ucb_c=self.ucb_c)
        self.children.append(child)
        return child

    def select(self) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        return max(self.children, key=self.get_ucb)

    def get_ucb(self, child):
        # value_sum is in between -1 and 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        q_value = ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.ucb_c * math.sqrt(math.log(self.visit_count) / child.visit_count)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, game: Quoridor, params: MCTSParams):
        self.game = game
        self.params = params

    def select(self, node: Node) -> Node:
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        while not node.should_expand():
            node = node.select()
        return node

    def distance_value(self, game: Quoridor) -> float:
        """
        Return an evaluation of how good the position looks for the current player based on the distance to the
        target and the number of walls remaining.
        """
        current_player = game.current_player
        other_player = Player(1 - current_player)
        distance_reward = game.player_distance_to_target(current_player) - game.player_distance_to_target(other_player)
        wall_reward = game.board.get_walls_remaining(other_player) - game.board.get_walls_remaining(current_player)

        return distance_reward + wall_reward / 100.0

    def simulate(self, initial_game: Quoridor) -> float:
        """
        Simulate playing randomly the rest of the game until one wins or max_steps are reached
        """
        game = copy.deepcopy(initial_game)

        next_player = game.current_player

        for i in range(self.params.max_steps):
            if game.check_win(next_player):
                return -1

            if game.check_win(1 - next_player):
                return 1

            actions = game.get_valid_actions()
            action = random.choice(actions)
            game.step(action)

        return 0

    def evaluate(self, game: Quoridor) -> float:
        if self.params.eval == "rand_sim":
            return self.simulate(game)

        if self.params.eval == "dist":
            return self.distance_value(game)

        assert False, f"Unknown evaluation method: {self.params.eval}"

    def search(self, initial_game: Quoridor):
        root = Node(initial_game, ucb_c=self.params.c)
        good_player = initial_game.current_player

        for _ in range(self.params.n):
            # Traverse down the tree guided by maximum UCB until we find a node to expand
            node = self.select(root)

            if node.game.check_win(good_player):
                value = 100
            elif node.game.check_win(1 - good_player):
                value = -100
            else:
                node = node.expand()
                value = self.evaluate(node.game)

            node.backpropagate(value)

        return root.children


class MCTSAgent(Agent):
    def __init__(self, params=MCTSParams(), **kwargs):
        super().__init__()
        self.params = params

    def name(self):
        return self.params.nick if self.params.nick else "mcts"

    @classmethod
    def params_class(cls):
        return MCTSParams

    def start_game(self, game, player_id):
        self.player_id = player_id

    def _log_action(self, children: list[Node]):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        scores = {
            child.action_taken: child.value_sum / child.visit_count
            for child in children
            if child.action_taken is not None  # Just to make the linter happy
        }

        self.action_log.action_score_ranking(scores)

    def get_action(self, observation):
        observation = observation["observation"]

        game, _, _ = construct_game_from_observation(observation, self.player_id)
        mcts = MCTS(game, self.params)
        children = mcts.search(game)

        sorted_children = sorted(children, key=lambda x: x.visit_count, reverse=True)
        top_children = sorted_children[: min(5, len(sorted_children))]

        self._log_action(top_children)
        return game.action_encoder.action_to_index(top_children[0].action_taken)
