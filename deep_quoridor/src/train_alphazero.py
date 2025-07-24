import time

import torch
import torch.multiprocessing as mp
from agents.alphazero.multiprocess_evaluator import EvaluatorClient, EvaluatorServer
from agents.alphazero.nn_evaluator import NNEvaluator
from quoridor import ActionEncoder
from utils import my_device


def _run_example_worker(worker_id, request_queue, result_queue, n):
    client = EvaluatorClient(worker_id, request_queue, result_queue)
    for ii in range(n):
        input_tensor = torch.Tensor((worker_id, ii))
        _ = client.evaluate(input_tensor)


def main():
    # Set multiprocessing start method to avoid tensor sharing issues and Mac bugs
    mp.set_start_method("spawn", force=True)

    # Game parameters
    board_size = 5
    max_walls = 3
    max_game_length = 200

    # Parallelism parameters
    batch_size = 5
    max_interbatch_time = 0.001
    num_workers = 12
    pool_size = 5
    n = 10000

    action_encoder = ActionEncoder(board_size)
    evaluator = NNEvaluator(action_encoder, my_device())

    request_queue = mp.SimpleQueue()
    result_queues = [mp.SimpleQueue() for _ in range(num_workers)]
    evaluator_server = EvaluatorServer(batch_size, max_interbatch_time, request_queue, result_queues)
    evaluator_server.start()

    workers = []
    for worker_id in range(num_workers):
        workers.append(
            mp.Process(target=_run_example_worker, args=(worker_id, request_queue, result_queues[worker_id], n))
        )

    t0 = time.time()

    for worker in workers:
        worker.start()

    t1 = time.time()

    for worker in workers:
        worker.join()

    t2 = time.time()

    evaluator_server.shutdown()
    evaluator_server.join()

    print(f"Worker startup time: {t1 - t0}")
    print(f"Total processing time {t2 - t0}")
    print(f"Throughput = {(n * num_workers) / (t2 - t0)}")
    print(evaluator_server.get_statistics())


if __name__ == "__main__":
    main()
