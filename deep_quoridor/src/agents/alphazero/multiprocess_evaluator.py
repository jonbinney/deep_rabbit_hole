"""
Wraps an evaluator for use by multiple client processes via Queues.
"""

import threading
import time
from collections import deque

import numpy as np
import torch
import torch.multiprocessing as mp
from quoridor import Quoridor
from utils import my_device

from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core.rotation import create_rotation_mapping


class EvaluatorClient:
    def __init__(self, client_id: int, request_queue: mp.SimpleQueue, result_queue: mp.SimpleQueue, board_size: int):
        self._client_id = client_id
        self._request_queue = request_queue
        self._result_queue = result_queue
        (self._action_mapping_original_to_rotated, self._action_mapping_rotated_to_original) = create_rotation_mapping(
            board_size
        )

    def evaluate(self, game: Quoridor):
        # Rotate the board if player 2 is playing so that we always work with player 1's perspective.
        game, is_board_rotated = self.rotate_if_needed_to_point_downwards(game)
        input_array = self.game_to_input_array(game)
        action_mask_array = game.get_action_mask()

        value, policy = self.evaluate_array(input_array, action_mask_array)

        if is_board_rotated:
            policy = self.rotate_policy_from_original(policy)

        return value, policy

    def evaluate_array(self, input_array: np.ndarray, action_mask_array: np.ndarray):
        self._request_queue.put((input_array, action_mask_array, self._client_id))
        value, policy = self._result_queue.get()
        return value, policy

    def rotate_policy_from_original(self, policy: np.ndarray):
        return policy[self._action_mapping_original_to_rotated]

    def rotate_if_needed_to_point_downwards(self, game: Quoridor):
        return NNEvaluator.rotate_if_needed_to_point_downwards(game)

    def game_to_input_array(self, game):
        return NNEvaluator.game_to_input_array(game)


class EvaluatorServer(threading.Thread):
    def __init__(
        self,
        evaluator: NNEvaluator,
        batch_size: int,
        max_interbatch_time: float,
        input_queue: mp.SimpleQueue,
        output_queues: list[mp.SimpleQueue],
        statistics_window_size=10000,
    ):
        super().__init__()
        self._evaluator = evaluator
        self._batch_size = batch_size
        self._max_interbatch_time = max_interbatch_time
        self._input_queue = input_queue
        self._output_queues = output_queues
        self._statistics_window_size = statistics_window_size
        self._statistics = deque(maxlen=statistics_window_size)
        self._shutdown = False

    def run(self):
        batch = []
        last_batch_end_time = time.time()

        while not self._shutdown:
            while len(batch) == 0:
                if self._shutdown:
                    break

                # If we used Queue instead of SimpleQueue, we could use a timeout in get(),
                # but that seems to cause deadlocks. Until we debug that, sticking to a
                # busy wait here.
                wait_until_time = last_batch_end_time + self._max_interbatch_time
                while time.time() < wait_until_time and self._input_queue.empty():
                    time.sleep(0.00001)

                if self._input_queue.empty():
                    break

                # Grab any more available inputs up to batch size
                while not self._input_queue.empty() and len(batch) < self._batch_size:
                    input_array, action_mask_array, client_id = self._input_queue.get()
                    batch.append((input_array, action_mask_array, client_id))

            if len(batch) > 0:
                stacked_input_tensor = torch.from_numpy(np.stack([x[0] for x in batch])).to(my_device())
                stacked_action_masks_tensor = torch.from_numpy(np.stack([x[1] for x in batch])).to(my_device())
                client_ids = [x[2] for x in batch]

                # Run the evaluator on all the inputs at once
                evaluation_start_time = time.time()
                stacked_values, stacked_policies = self._evaluator.batch_evaluate(
                    stacked_input_tensor, stacked_action_masks_tensor
                )
                evaluation_end_time = time.time()

                # Send the results back to the clients
                for row, client_id in enumerate(client_ids):
                    self._output_queues[client_id].put((stacked_values[row], stacked_policies[row]))

                # Keep some statistics for tuning and debugging
                self._statistics.append((len(batch), evaluation_start_time, evaluation_end_time))

                # Clear out everything we just computed
                batch = []

            last_batch_end_time = time.time()

    def get_statistics(self):
        batch_size_sum = 0
        evaluation_time_sum = 0
        num_batches = 0
        for batch_size, evaluation_start_time, evaluation_end_time in self._statistics:
            batch_size_sum += batch_size
            evaluation_time_sum += evaluation_end_time - evaluation_start_time
            num_batches += 1
        if num_batches > 0:
            return {
                "average_batch_size": batch_size_sum / num_batches,
                "average_evaluation_time": evaluation_time_sum / num_batches,
            }
        else:
            return {}

    def shutdown(self):
        self._shutdown = True
