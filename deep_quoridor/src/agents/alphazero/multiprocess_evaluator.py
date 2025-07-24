"""
Wraps an evaluator for use by multiple client processes via Queues.
"""

import threading
import time
from collections import deque

import torch
import torch.multiprocessing as mp


class EvaluatorClient:
    def __init__(self, client_id: int, request_queue: mp.SimpleQueue, result_queue: mp.SimpleQueue):
        self._client_id = client_id
        self._request_queue = request_queue
        self._result_queue = result_queue

    def evaluate(self, input_tensor: torch.Tensor):
        # Convert tensor to regular Python data for safe IPC
        input_data = input_tensor.detach().cpu().numpy().tolist()
        self._request_queue.put((input_data, self._client_id))
        result_data = self._result_queue.get()
        # Convert back to tensor
        return torch.tensor(result_data)


class EvaluatorServer(threading.Thread):
    def __init__(
        self,
        batch_size: int,
        max_interbatch_time: float,
        input_queue: mp.SimpleQueue,
        output_queues: list[mp.SimpleQueue],
        statistics_window_size=10000,
    ):
        super().__init__()
        self._batch_size = batch_size
        self._max_interbatch_time = max_interbatch_time
        self._input_queue = input_queue
        self._output_queues = output_queues
        self._statistics_window_size = statistics_window_size
        self._statistics = deque(maxlen=statistics_window_size)
        self._shutdown = False

    def run(self):
        last_batch_end_time = time.time()
        while not self._shutdown:
            input_tensors = []
            client_ids = []
            while len(input_tensors) < self._batch_size:
                if self._shutdown:
                    break

                wait_until_time = last_batch_end_time + self._max_interbatch_time
                while time.time() < wait_until_time and self._input_queue.empty():
                    # If we used Queue instead of SimpleQueue, we could use a timeout in get(),
                    # but that seems to cause deadlocks.
                    time.sleep(0.001)

                if self._input_queue.empty():
                    break

                input_data, client_id = self._input_queue.get()
                # Convert back to tensor
                input_tensor = torch.tensor(input_data)
                input_tensors.append(input_tensor)
                client_ids.append(client_id)

            if len(input_tensors) > 0:
                stacked_input_tensor = torch.stack(input_tensors)
                evaluation_start_time = time.time()
                stacked_output_tensor = stacked_input_tensor * 1
                evaluation_end_time = time.time()

                for row, client_id in enumerate(client_ids):
                    output_tensor = stacked_output_tensor[row]
                    # Convert tensor to regular Python data for safe IPC
                    output_data = output_tensor.detach().cpu().numpy().tolist()
                    self._output_queues[client_id].put(output_data)

                    self._statistics.append((len(input_tensors), evaluation_start_time, evaluation_end_time))

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
