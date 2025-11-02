from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import dirichlet
from quoridor import Quoridor


class QuoridorKey:
    def __init__(self, game: Quoridor):
        self.game = game
        self.hash = game.get_fast_hash()

    def __hash__(self):
        return self.hash

    def __eq__(self, other: "QuoridorKey"):
        return self.hash == other.hash and self.game == other.game


@dataclass
class NodeConfig:
    action_size: int
    c_puct: float


class Node:
    class Children:
        def __init__(self, priors: np.ndarray):
            indices = np.flatnonzero(priors)
            n = len(indices)
            self.priors = priors[indices]
            self.actions = indices
            self.visit_counts = np.zeros((n), dtype=np.int32)
            self.value_sums = np.zeros((n), dtype=np.float32)
            self.children_nodes = [None] * n

    def __init__(self, game: Quoridor, parent: Optional["Node"], pos_at_parent: int, config: NodeConfig):
        self.game = game
        self.parent = parent
        self.pos_at_parent = pos_at_parent
        self.config = config

        self.children = None

    def should_expand(self):
        return self.children is None

    def expand(self, priors: np.ndarray):
        self.children = Node.Children(priors)

    def select(self) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        if self.parent is None and self.should_expand():
            return self  # TO DO,

        assert self.children
        children = self.children

        # Compute Q values: value_sums / visit_counts, but 0 where visit_counts == 0
        q_values = np.zeros_like(children.value_sums)
        nonzero_mask = children.visit_counts != 0
        q_values[nonzero_mask] = children.value_sums[nonzero_mask] / children.visit_counts[nonzero_mask]
        # print(f"{q_values=}")
        total_visit_count = np.sum(children.visit_counts) + 1
        ucb = q_values + children.priors * self.config.c_puct * np.sqrt(total_visit_count) / (children.visit_counts + 1)

        # print(f"{ucb=}")
        max_index = int(np.argmax(ucb))
        if children.children_nodes[max_index] is None:
            game = self.game.create_new()
            action = self.game.action_encoder.index_to_action(children.actions[max_index])
            game.step(action)
            children.children_nodes[max_index] = Node(game, self, max_index, self.config)

        # assert self.children_nodes[max_index] is not None
        return children.children_nodes[max_index]

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        # TODo not recursive
        if self.parent is not None:
            children = self.parent.children
            assert children
            children.value_sums[self.pos_at_parent] += value
            children.visit_counts[self.pos_at_parent] += 1
            self.parent.backpropagate(-value)

    def backpropagate_result(self, value: float):
        return self.backpropagate(value)  # TO DO

    def results(self):
        assert self.children
        results = np.zeros((self.config.action_size), dtype=np.int32)
        indices = np.nonzero(self.children.visit_counts)
        results[self.children.actions[indices]] = self.children.visit_counts[indices]
        # for i, child in enumerate(self.children_nodes):
        #     if child is None:
        #         continue

        #     results[self.actions[i]] = self.visit_counts[i]

        return results

    def value(self):
        return -(np.sum(self.children.value_sums) / np.sum(self.children.visit_counts))


class MCTS:
    def __init__(
        self,
        n: Optional[int],
        k: Optional[int],
        ucb_c: float,
        noise_epsilon: float,
        noise_alpha: float,
        max_steps: int,  # -1 means no limit
        evaluator,
        visited_states: set,
    ):
        assert n is not None or k is not None, "Either n or k need to be specified"
        self.n = n
        self.k = k
        self.ucb_c = ucb_c
        self.noise_epsilon = noise_epsilon
        self.noise_alpha = noise_alpha
        self.max_steps = max_steps
        self.evaluator = evaluator
        self.visited_states = visited_states

    def select(self, node: Node) -> Node:
        """
        Select a node to expand, starting from the given node.
        As long as the passed node has leaves to expand, that one will be selected.
        Otherwise, if the node is fully expanded, then its best child will be selected.
        """
        while not node.should_expand():
            node = node.select()
        return node

    def search_batch(self, initial_games: list[Quoridor]):
        if self.k is not None:
            num_iterations = [self.k * int(np.sum(np.array(game.get_action_mask()) == 1)) for game in initial_games]
        else:
            assert self.n is not None, "n must be specified if k is not"
            num_iterations = [self.n for _ in initial_games]

        max_iterations = max(num_iterations)

        # TODO: redo MCTS 0
        config = NodeConfig(initial_games[0].action_encoder.num_actions, self.ucb_c)
        roots = [Node(g, None, 0, config) for g in initial_games]

        # # When n is 0, it plays just with the NN and doesn't actually perform MCTS.
        # # For this, we just set the visit counts to a value proportional to the prior
        if self.n == 0:
            value_batch, priors_batch = self.evaluator.evaluate_batch([node.game for node in roots])
            # for root, value, priors in zip(roots, value_batch, priors_batch):
            #     # TODO maybe I don't need to do this
            #     root.expand(priors)
            #     root.backpropagate(-value)

            #     visit_counts_list = [(p * 1000).astype(np.int32) for p in priors]

            # return visit_counts_list, [root.value() for root in roots]

            return [[(p * 1000).astype(np.int32) for p in priors] for priors in priors_batch], [
                -value
                for value in value_batch  # value or -value?
            ]

        for iteration in range(max_iterations):
            need_evaluation = []  # (root, node)
            for game_idx, root in enumerate(roots):
                # The number of iterations may be less if using mcts_k
                if iteration >= num_iterations[game_idx]:
                    continue

                # Traverse down the tree guided by maximum UCB until we find a node to expand
                node = self.select(root)

                if node.game.is_game_over():
                    # The player who just made a move must have won.
                    node.backpropagate_result(1)
                elif self.max_steps >= 0 and node.game.completed_steps >= self.max_steps:
                    node.backpropagate_result(0)
                else:
                    need_evaluation.append((root, node))

            games_to_evaluate = [node.game for _, node in need_evaluation]
            value_batch, priors_batch = self.evaluator.evaluate_batch(games_to_evaluate)

            for (root, node), value, priors in zip(need_evaluation, value_batch, priors_batch):
                if node is root and self.noise_epsilon > 0.0:
                    # To encourage exploration we add noise to the priors at the root node.
                    # NOTE: It isn't clear from the paper whether we should apply the dirichlet noise
                    # only to the valid actions, or whether we should apply it to all actions and then
                    # re-mask and re-normalize. These might work out to be equivalent, but I'm not sure of
                    # that either. For now we apply the noise _only_ to the valid actions. That seems to
                    # match what the Openspiel python implementation does.
                    (valid_action_indices,) = np.nonzero(priors > 0.0)
                    dirichlet_noise = dirichlet([self.noise_alpha] * len(valid_action_indices))
                    priors[valid_action_indices] *= 1.0 - self.noise_epsilon
                    priors[valid_action_indices] += self.noise_epsilon * dirichlet_noise
                node.expand(priors)
                node.backpropagate(-value)

        # Negate the value because the value is actually the value that the opponent
        # got for getting to that state.
        # return [root.children for root in roots], [-(root.value_sum / root.visit_count) for root in roots]
        return [root.results() for root in roots], [root.value() for root in roots]

    def search(self, initial_game: Quoridor):
        children_batch, root_value_batch = self.search_batch([initial_game])
        return children_batch[0], root_value_batch[0]
