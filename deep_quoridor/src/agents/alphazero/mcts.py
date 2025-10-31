from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.random import dirichlet
from quoridor import Action, Quoridor


class QuoridorKey:
    def __init__(self, game: Quoridor):
        self.game = game
        self.hash = game.get_fast_hash()

    def __hash__(self):
        return self.hash

    def __eq__(self, other: "QuoridorKey"):
        return self.hash == other.hash and self.game == other.game


@dataclass
class NodeProConfig:
    action_size: int
    c_puct: float


class NodePro:
    def __init__(
        self, game: Quoridor, parent: Optional["NodePro"], pos_at_parent: int, config: Optional[NodeProConfig] = None
    ):
        self.game = game
        self.parent = parent
        self.pos_at_parent = pos_at_parent
        if self.parent:
            self.config = self.parent.config
        else:
            assert config is not None
            self.config = config

        self.actions = None
        self.visit_counts = None
        self.value_sums = None
        self.priors = None
        self.children_nodes = None
        self.expanded = False

    def should_expand(self):
        return not self.expanded

    def expand(self, priors: np.ndarray):
        indices = np.flatnonzero(priors)
        n = len(indices)
        self.priors = priors[indices]
        self.actions = indices
        self.visit_counts = np.zeros((n), dtype=np.int32)
        self.value_sums = np.zeros((n), dtype=np.float32)
        self.children_nodes: list[Optional[NodePro]] = [None] * n
        self.expanded = True

    def select(self) -> "NodePro":
        """
        Return the child of the current node with the highest ucb
        """
        if self.parent is None and self.should_expand():
            return self  # TO DO,

        assert self.expanded and self.value_sums and self.visit_counts and self.priors and self.children_nodes

        # Compute Q values: value_sums / visit_counts, but 0 where visit_counts == 0
        q_values = np.zeros_like(self.value_sums)
        nonzero_mask = self.visit_counts != 0
        q_values[nonzero_mask] = self.value_sums[nonzero_mask] / self.visit_counts[nonzero_mask]
        # print(f"{q_values=}")
        total_visit_count = np.sum(self.visit_counts) + 1
        ucb = q_values + self.priors * self.config.c_puct * np.sqrt(total_visit_count) / (self.visit_counts + 1)

        # print(f"{ucb=}")
        max_index = int(np.argmax(ucb))
        if self.children_nodes[max_index] is None:
            game = self.game.create_new()
            action = self.game.action_encoder.index_to_action(self.actions[max_index])
            game.step(action)
            self.children_nodes[max_index] = NodePro(game, self, max_index)

        # assert self.children_nodes[max_index] is not None
        return self.children_nodes[max_index]

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        # TODo not recursive
        if self.parent is not None:
            assert self.parent.value_sums and self.parent.visit_counts
            self.parent.value_sums[self.pos_at_parent] += value
            self.parent.visit_counts[self.pos_at_parent] += 1
            self.parent.backpropagate(-value)

    def backpropagate_result(self, value: float):
        return self.backpropagate(value)  # TO DO

    def results_dict(self):
        results = {}
        for i, child in enumerate(self.children_nodes):
            if child is None:
                continue

            results[self.actions[i]] = self.visit_counts[i]

        return results

    def results(self):
        results = np.zeros((self.config.action_size), dtype=np.int32)
        for i, child in enumerate(self.children_nodes):
            if child is None:
                continue

            results[self.actions[i]] = self.visit_counts[i]

        return results

    def children(self):
        children = []
        for i, child in enumerate(self.children_nodes):
            if child is None:
                continue
            action = self.game.action_encoder.index_to_action(self.actions[i])

            n = Node(child.game, None, action, 0.0, self.priors[i])  # TO DO
            n.visit_count = self.visit_counts[i]
            n.value_sum = self.value_sums[i]
            children.append(n)
            # print(f"===== {n.action_taken}, {n.visit_count}, {n.value_sum}")

        return children

    def value(self):
        return -(np.sum(self.value_sums) / np.sum(self.visit_counts))


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

    def select(self, visited_states) -> "Node":
        """
        Return the child of the current node with the highest ucb
        """
        ucbc_visitcount = self.ucb_c * np.sqrt(self.visit_count)

        def get_child_ucb(child: "Node") -> float:
            """
            Calculate the UCB value for a child node.
            """
            penalized = -1 if visited_states and QuoridorKey(child.game) in visited_states else 0
            q_value = 0.0
            if child.visit_count != 0:
                q_value = child.value_sum / child.visit_count
            return q_value + child.prior * ucbc_visitcount / (child.visit_count + 1) + penalized

        return max(self.children, key=get_child_ucb)

    def backpropagate(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        """
        self.value_sum += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backpropagate(-value)

    def backpropagate_result(self, value: float):
        """
        Update the nodes from the current node up to the tree by increasing the visit count and adding the value
        It also tracks actual game results (wins and losses)
        """
        self.value_sum += value
        self.visit_count += 1
        self.wins = self.wins + (1 if value == 1 else 0)
        self.losses = self.losses + (1 if value == -1 else 0)

        if self.parent is not None:
            self.parent.backpropagate_result(-value)


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

    def select(self, node: NodePro) -> NodePro:
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
        config = NodeProConfig(initial_games[0].action_encoder.num_actions, self.ucb_c)
        roots = [NodePro(g, None, 0, config) for g in initial_games]

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
