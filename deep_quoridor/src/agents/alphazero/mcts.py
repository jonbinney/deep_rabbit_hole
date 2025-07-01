import copy
from typing import Optional

import numpy as np
from quoridor import Action, ActionEncoder, Quoridor


class Node:
    def __init__(
        self,
        game: Quoridor,
        parent: Optional["Node"] = None,
        action_taken: Optional[Action] = None,
        ucb_c: float = 1.0,
        prior: float = 0.0,
    ):
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        self.ucb_c = ucb_c
        self.prior = prior

        if game is not None:
            self.action_encoder = ActionEncoder(game.board.board_size)

    def should_expand(self):
        return len(self.children) == 0

    def expand(self, priors: np.ndarray):
        """
        Create all the children of the current node.
        """
        for action_index, prob in enumerate(priors):
            if prob < 0.0 or prob > 1.0:
                raise ValueError("Invalid action probability in policy")

            action = self.action_encoder.index_to_action(action_index)

            if prob > 0.0:
                # Apply the action.
                game = copy.deepcopy(self.game)
                game.step(action)

                child = Node(game, parent=self, action_taken=action, ucb_c=self.ucb_c, prior=prob)
                self.children.append(child)

    def select(self) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        child_ucbs = self.get_child_ucbs()
        child_i = np.argmax(child_ucbs)
        return self.children[child_i]

    def get_child_ucbs(self):
        visit_counts = np.array([child.visit_count for child in self.children])
        value_sums = np.array([child.value_sum for child in self.children])
        priors = np.array([child.prior for child in self.children])

        # value_sum is in between -1 and 1, so doing (avg + 1) / 2 would make it in the range [0, 1]
        q_values = (np.divide(value_sums, visit_counts, where=visit_counts != 0) + 1) / 2

        return q_values + self.ucb_c * priors * np.sqrt(visit_counts) / (visit_counts + 1)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    def __init__(self, n: int, ucb_c: float, evaluator):
        self.n = n
        self.ucb_c = ucb_c
        self.evaluator = evaluator

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
                value, priors = self.evaluator.evaluate(node.game)
                node.expand(priors)

            node.backpropagate(value)

        return root.children
