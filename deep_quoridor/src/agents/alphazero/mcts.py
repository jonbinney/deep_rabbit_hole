from typing import Optional

import numpy as np
from quoridor import Action, Quoridor


class Node:
    def __init__(
        self,
        game: Optional[Quoridor] = None,  # If None, the game will be computed lazily from the parent when needed
        parent: Optional["Node"] = None,
        action_taken: Optional[Action] = None,
        ucb_c: float = 1.0,
        prior: float = 0.0,
        node_id: tuple[int] = (),  # Sequence of action id's to go from root to here, which acts as a unique id
    ):
        if game is None:
            assert node_id is not None, "When constructing a node without a game, the node_id must be provided"
            assert parent is not None, "When constructing a node without a game, the parent should be provided"
            assert action_taken is not None, (
                "When constructing a node without a game, the action_taken should be provided"
            )

        self.node_id = node_id
        self._game = game
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.visit_count = 0
        self.value_sum = 0.0

        self.ucb_c = ucb_c
        self.prior = prior

    def __hash__(self):
        return hash(self.node_id)

    def __eq__(self, other):
        return self.node_id == other.node_id

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
                child_node_id = self.node_id + (action_index,)
                child = Node(
                    node_id=child_node_id, game=None, parent=self, action_taken=action, ucb_c=self.ucb_c, prior=prob
                )
                self.children.append(child)

    def select(self) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        child_ucbs = self.get_child_ucbs()
        best_child_i = np.argmax(child_ucbs)

        return self.children[best_child_i]

    def get_child_ucbs(self):
        ucbc_visitcount = self.ucb_c * np.sqrt(self.visit_count)

        def get_child_ucb(child: "Node") -> float:
            """
            Calculate the UCB value for a child node.
            """
            q_value = 0.5
            if child.visit_count != 0:
                q_value = (child.value_sum / child.visit_count + 1) / 2
            return q_value + child.prior * ucbc_visitcount / (child.visit_count + 1)

        return [get_child_ucb(child) for child in self.children]

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
        root = Node(game=initial_game, ucb_c=self.ucb_c)

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

    def batch_search(self, initial_game: Quoridor, batch_size: int):
        root = Node(initial_game, ucb_c=self.ucb_c)

        batch = set()
        nodes_to_search = [root] * self.n
        while len(nodes_to_search) > 0:
            # Traverse down the tree guided by maximum UCB until we find a node to expand
            start_node = nodes_to_search.pop()
            node = self.select(start_node)

            if node.game.is_game_over():
                # The player who just made a move must have won.
                value = 1
                node.backpropagate(value)
            # TODO: Handle ties (these happen when arena decides game is taking too long)
            elif node in batch:
                # If we keep finding the same node when searching, e.g. if we have only the root
                # node in the tree or one search path is very likely, we could end up in a deadlock
                # where we never get a batch big enough to expand. To avoid this, when we see a node
                # already in the batch, we immediately expand the entire batch and put that node back
                # on the stack (we found it twice; the first time resulted in the expand and the second
                # select needs to be finished now that the node is expanded.
                self.expand_batch(list(batch))
                batch.clear()
                nodes_to_search.append(node)
            elif len(batch) >= batch_size:
                # Batch is large enough, expand the nodes
                self.expand_batch(list(batch))
                batch.clear()
            else:
                batch.add(node)

        return root.children

    def expand_batch(self, batch: list[Node]):
        print(f"Batch size: {len(batch)}")
        values, policies = self.evaluator.evaluate_batch([node.game for node in batch])
        for node, value, policy in zip(batch, values, policies):
            print(
                f"Expanding {(batch[0].node_id)}: visit_count={batch[0].visit_count} num_children={len(batch[0].children)}"
            )
            node.expand(policy)
            node.backpropagate(value)
