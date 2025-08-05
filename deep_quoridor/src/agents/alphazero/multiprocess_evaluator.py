"""
Wraps an evaluator for use by multiple client processes via Queues.
"""

import queue
import threading
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from quoridor import Player, Quoridor
from utils import my_device

from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core.rotation import create_rotation_mapping


class EvaluatorClient:
    def __init__(
        self,
        board_size: int,
        client_id: int,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
    ):
        self._client_id = client_id
        self._request_queue = request_queue
        self._result_queue = result_queue
        (self._action_mapping_original_to_rotated, self._action_mapping_rotated_to_original) = create_rotation_mapping(
            board_size
        )
        # byte_representation_of_game_state -> (, value, policy)
        self._cache = {}

    def evaluate(self, game: Quoridor, extra_games_to_evaluate: list[Quoridor] = []):
        if game.get_byte_repr() in self._cache:
            return self._cache[game.get_byte_repr()]

        all_games = [game] + extra_games_to_evaluate
        all_hashes = np.stack([g.get_byte_repr() for g in all_games])
        all_games = [NNEvaluator.rotate_if_needed_to_point_downwards(g)[0] for g in all_games]
        all_games_inputs_array = np.stack([NNEvaluator.game_to_input_array(g) for g in all_games])
        all_games_action_masks = np.stack([g.get_action_mask() for g in all_games])

        values, policies = self.evaluate_array(all_games_inputs_array, all_games_action_masks)

        for game_i, game in enumerate(all_games):
            # Sanity checks
            assert np.all(policies[game_i] >= 0), "Policy contains negative probabilities"
            assert np.abs(np.sum(policies[game_i]) - 1) < 1e-6, "Policy does not sum to 1"
            assert np.isfinite(values[game_i]), "Policy or value is non-finite"

            # If the game was originally rotated, rotate the resulting back
            if game.get_current_player() == Player.TWO:
                policies[game_i] = policies[game_i][self._action_mapping_rotated_to_original]

            self._cache[all_hashes[game_i]] = (values[game_i], policies[game_i])

        return values[0], policies[0]

    def evaluate_array(self, input_arrays: np.ndarray, action_mask_arrays: np.ndarray):
        self._request_queue.put((input_arrays, action_mask_arrays, self._client_id))
        value, policy = self._result_queue.get(timeout=10)
        return value, policy

    def rotate_policy_from_original(self, policy: np.ndarray):
        return policy[self._action_mapping_original_to_rotated]

    def rotate_if_needed_to_point_downwards(self, game: Quoridor):
        return NNEvaluator.rotate_if_needed_to_point_downwards(game)

    def game_to_input_array(self, game):
        return NNEvaluator.game_to_input_array(game)

    def clear_cache(self):
        self._cache.clear()


class EvaluatorServer(threading.Thread):
    def __init__(
        self,
        evaluator: NNEvaluator,
        input_queue: mp.Queue,
        output_queues: list[mp.Queue],
        statistics_window_size=1000000000,
    ):
        super().__init__()
        self._evaluator = evaluator
        self._cache = {}
        self._input_queue = input_queue
        self._output_queues = output_queues
        self._statistics_window_size = statistics_window_size
        self._statistics = deque(maxlen=statistics_window_size)
        self._shutdown = False

    def run(self):
        while not self._shutdown:
            try:
                inputs_array, action_masks_array, client_id = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            inputs_tensor = torch.from_numpy(inputs_array).to(my_device())
            action_masks_tensor = torch.from_numpy(action_masks_array).to(my_device())

            evaluation_start_time = time.time()
            values_tensor, policies_tensor = self._evaluator.evaluate_tensors(inputs_tensor, action_masks_tensor)
            evaluation_end_time = time.time()

            # Transfer policies and values back to CPU and turn them into arrays
            policies_array = policies_tensor.cpu().numpy()
            values_array = values_tensor.cpu().flatten().numpy()

            self._output_queues[client_id].put((values_array, policies_array))

            self._statistics.append((evaluation_start_time, evaluation_end_time))

    def train_prepare(self, *args, **kwargs):
        return self._evaluator.train_prepare(*args, **kwargs)

    def train_iteration(self, *args, **kwargs):
        self._cache.clear()
        return self._evaluator.train_iteration(*args, **kwargs)

    def get_statistics(self):
        evaluation_time_sum = 0
        num_evaluations = len(self._statistics)
        first_evaluation_time = None
        last_evaluation_time = None
        for evaluation_start_time, evaluation_end_time in self._statistics:
            if first_evaluation_time is None:
                first_evaluation_time = evaluation_start_time
            last_evaluation_time = evaluation_end_time
            evaluation_time_sum += evaluation_end_time - evaluation_start_time

        if num_evaluations > 0:
            if first_evaluation_time is None or last_evaluation_time is None:
                duty_cycle = None
            else:
                duty_cycle = evaluation_time_sum / (last_evaluation_time - first_evaluation_time)

            return {
                "duty_cycle": duty_cycle,
                "average_evaluation_time": evaluation_time_sum / num_evaluations,
                "num_evaluations": num_evaluations,
            }
        else:
            return {}

    def shutdown(self):
        self._shutdown = True
