"""
Wraps an evaluator for use by multiple client processes via Queues.
"""

import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp
from quoridor import Player, Quoridor
from utils import my_device

from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core.rotation import create_rotation_mapping


@dataclass
class EvaluationInfo:
    """
    Information about one call to the evaluator.
    """

    evaluation_start_time: float
    evaluation_end_time: float
    num_games_evaluated: int
    used_cache: bool


@dataclass
class EvaluatorStatistics:
    """
    Statistics about all evaluations since this evaluator was created.
    """

    duty_cycle: float
    average_evaluation_time: float
    num_evaluations: int
    num_cache_hits: int


def compute_statistics(evaluation_log: list[EvaluationInfo]) -> Optional[EvaluatorStatistics]:
    evaluation_time_sum = 0
    num_evaluations = len(evaluation_log)
    num_cache_hits = 0
    first_evaluation_time = None
    last_evaluation_time = None
    for eval_info in evaluation_log:
        if first_evaluation_time is None:
            first_evaluation_time = eval_info.evaluation_start_time
        last_evaluation_time = eval_info.evaluation_end_time
        evaluation_time_sum += eval_info.evaluation_end_time - eval_info.evaluation_start_time
        if eval_info.used_cache:
            num_cache_hits += 1

    if num_evaluations > 0:
        if first_evaluation_time is None or last_evaluation_time is None:
            duty_cycle = None
        else:
            duty_cycle = evaluation_time_sum / (last_evaluation_time - first_evaluation_time)

        return EvaluatorStatistics(duty_cycle, evaluation_time_sum / num_evaluations, num_evaluations, num_cache_hits)
    else:
        return None


class EvaluatorClient:
    def __init__(
        self,
        board_size: int,
        client_id: int,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        max_log_len=int(1e8),
    ):
        self._client_id = client_id
        self._request_queue = request_queue
        self._result_queue = result_queue
        (self._action_mapping_original_to_rotated, self._action_mapping_rotated_to_original) = create_rotation_mapping(
            board_size
        )
        self._cache = {}
        self._evaluation_log = deque(maxlen=max_log_len)

    def evaluate(self, game: Quoridor, extra_games_to_evaluate: list[Quoridor] = []):
        evaluation_start_time = time.time()

        game, _ = self.rotate_if_needed_to_point_downwards(game)
        input_array = self.game_to_input_array(game)
        cache_key = input_array.tobytes()
        if cache_key in self._cache:
            cached_value, cached_policy, _ = self._cache[cache_key]
            if game.is_rotated():
                cached_policy = cached_policy[self._action_mapping_rotated_to_original]
            evaluation_end_time = time.time()
            self._evaluation_log.append(EvaluationInfo(evaluation_start_time, evaluation_end_time, 1, True))
            return cached_value, cached_policy

        all_games = [game] + [self.rotate_if_needed_to_point_downwards(g)[0] for g in extra_games_to_evaluate]
        all_games_inputs_array = np.stack([input_array] + [NNEvaluator.game_to_input_array(g) for g in all_games[1:]])
        all_games_action_masks = np.stack([g.get_action_mask() for g in all_games])

        values_array, policies_array = self.evaluate_arrays(all_games_inputs_array, all_games_action_masks)

        for row_i in range(len(values_array)):
            # Sanity checks
            assert np.all(policies_array[row_i] >= 0), "Policy contains negative probabilities"
            assert np.abs(np.sum(policies_array[row_i]) - 1) < 1e-6, "Policy does not sum to 1"
            assert np.isfinite(values_array[row_i]), "Policy or value is non-finite"

            # Update our local cache
            cache_key = all_games_inputs_array[row_i].tobytes()
            self._cache[cache_key] = (values_array[row_i], policies_array[row_i], game.copy())

        evaluation_end_time = time.time()
        self._evaluation_log.append(EvaluationInfo(evaluation_start_time, evaluation_end_time, 1, False))

        value = values_array[0]
        policy = policies_array[0]
        if game.is_rotated():
            policy = policy[self._action_mapping_rotated_to_original]

        return value, policy

    def evaluate_arrays(self, input_arrays: np.ndarray, action_mask_arrays: np.ndarray):
        self._request_queue.put((input_arrays, action_mask_arrays, self._client_id))
        values_array, policies_array = self._result_queue.get()
        return values_array, policies_array

    def rotate_policy_from_original(self, policy: np.ndarray):
        return policy[self._action_mapping_original_to_rotated]

    def rotate_if_needed_to_point_downwards(self, game: Quoridor):
        return NNEvaluator.rotate_if_needed_to_point_downwards(game)

    def game_to_input_array(self, game):
        return NNEvaluator.game_to_input_array(game)

    def clear_cache(self):
        self._cache.clear()

    def get_statistics(self):
        return compute_statistics(self._evaluation_log)


class EvaluatorServer(threading.Thread):
    def __init__(
        self,
        evaluator: NNEvaluator,
        input_queue: mp.Queue,
        output_queues: list[mp.Queue],
        max_log_len=int(1e8),
    ):
        super().__init__()
        self._evaluator = evaluator
        self._cache = {}
        self._input_queue = input_queue
        self._output_queues = output_queues
        self._max_l = max_log_len
        self._evaluation_log = deque(maxlen=max_log_len)
        self._shutdown = False

    def run(self):
        while not self._shutdown:
            try:
                inputs_array, action_masks_array, client_id = self._input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            values_array, policies_array = self._evaluate_arrays(inputs_array, action_masks_array)
            self._send_result_to_client(client_id, values_array, policies_array)

            # Update the cache
            for row_i in range(len(values_array)):
                self._cache[inputs_array[row_i].tobytes()] = (values_array[row_i], policies_array[row_i])

    def _evaluate_arrays(self, inputs_array, action_masks_array):
        evaluation_start_time = time.time()
        used_cache = False

        # If the first input array is in our cache, we return the cached value, policy for that
        # one and don't evaluate the others.
        cache_key = inputs_array[0].tobytes()
        if cache_key in self._cache:
            value, policy = self._cache.get(cache_key)
            values_array = np.expand_dims(value, 0)
            policies_array = np.expand_dims(policy, 0)
            used_cache = True
        else:
            inputs_tensor = torch.from_numpy(inputs_array).to(my_device())
            action_masks_tensor = torch.from_numpy(action_masks_array).to(my_device())

            # Do the actual evalution
            values_tensor, policies_tensor = self._evaluator.evaluate_tensors(inputs_tensor, action_masks_tensor)

            values_array = values_tensor.cpu().flatten().numpy()
            policies_array = policies_tensor.cpu().numpy()

        evaluation_end_time = time.time()
        self._evaluation_log.append(
            EvaluationInfo(evaluation_start_time, evaluation_end_time, len(inputs_array), used_cache)
        )
        return values_array, policies_array

    def _send_result_to_client(self, client_id, values_array, policies_array):
        self._output_queues[client_id].put((values_array, policies_array))

    def train_prepare(self, *args, **kwargs):
        return self._evaluator.train_prepare(*args, **kwargs)

    def train_iteration(self, *args, **kwargs):
        self._cache.clear()
        return self._evaluator.train_iteration(*args, **kwargs)

    def get_statistics(self):
        return compute_statistics(self._evaluation_log)

    def shutdown(self):
        self._shutdown = True
