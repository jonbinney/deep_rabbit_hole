from typing import Optional

import numpy as np
from quoridor import Action, Quoridor


class Node:
    def __init__(
        self,
        game: Optional[Quoridor],  # If None, the game will be computed lazily from the parent when needed
        parent: Optional["Node"] = None,
        action_taken: Optional[Action] = None,
        ucb_c: float = 1.0,
        prior: float = 0.0,
    ):
        if game is None:
            assert parent is not None, "When constructing a node without a game, the parent should be provided"
            assert action_taken is not None, (
                "When constructing a node without a game, the action_taken should be provided"
            )

        self._game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0
        self.wins = 0
        self.losses = 0

        self.ucb_c = ucb_c
        self.prior = prior

    @property
    def game(self) -> Quoridor:
        """
        The game is lazily set when used, by copying the parent's game and applying the action taken.
        """
        if self._game is None:
            assert self.parent is not None, "The root node should have a non-None game attribute"
            assert self.action_taken is not None, "The root node should have a non-None action_taken attribute"
            self._game = self.parent.game.create_new()
            self._game.step(self.action_taken)

        return self._game

    def should_expand(self):
        return len(self.children) == 0

    def expand(self, priors: np.ndarray):
        """
        Create all the children of the current node.
        """
        for action_index, prob in enumerate(priors):
            assert 0.0 <= prob <= 1.0, "Invalid action probability in policy"
            if prob > 0.0:
                action = self.game.action_encoder.index_to_action(action_index)
                child = Node(game=None, parent=self, action_taken=action, ucb_c=self.ucb_c, prior=prob)
                self.children.append(child)

    def select(self) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        ucbc_visitcount = self.ucb_c * np.sqrt(self.visit_count)

        def get_child_ucb(child: "Node") -> float:
            """
            Calculate the UCB value for a child node.
            """
            q_value = 0.5
            if child.visit_count != 0:
                q_value = (child.value_sum / child.visit_count + 1) / 2
            return q_value + child.prior * ucbc_visitcount / (child.visit_count + 1)

        return max(self.children, key=get_child_ucb)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1
        self.wins = self.wins + 1 if value == 1 else 0
        self.losses = self.losses + 1 if value == -1 else 0

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, n: int, ucb_c: float, evaluator, pre_evaluate_nodes_total: int = 64):
        self.n = n
        self.ucb_c = ucb_c
        self.evaluator = evaluator
        self.new_nodes = []
        self.extra_eval = pre_evaluate_nodes_total - 1

    def select(self, node: Node) -> Node:
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        while not node.should_expand():
            node = node.select()
        return node

    def search(self, initial_game: Quoridor):
        root = Node(initial_game, ucb_c=self.ucb_c)

        for _ in range(self.n):
            # Traverse down the tree guided by maximum UCB until we find a node to expand
            node = self.select(root)

            if node.game.is_game_over():
                # The player who just made a move must have won.
                value = 1
            # TODO: Handle ties (these happen when arena decides game is taking too long)
            else:
                games_to_evaluate = [n.game for n in self.new_nodes[: self.extra_eval]]
                self.new_nodes = self.new_nodes[self.extra_eval :]
                value, priors = self.evaluator.evaluate(node.game, games_to_evaluate)
                node.expand(priors)
                self.new_nodes.extend(node.children)

            node.backpropagate(value)

        return root.children
