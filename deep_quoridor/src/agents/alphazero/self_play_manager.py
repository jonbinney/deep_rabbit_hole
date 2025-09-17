import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import quoridor_env
from quoridor import ActionEncoder
from utils import my_device, parse_subargs, set_deterministic

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.multiprocess_evaluator import EvaluatorClient, EvaluatorServer, EvaluatorStatistics
from agents.alphazero.nn_evaluator import NNEvaluator


@dataclass
class GameParams:
    board_size: int
    max_walls: int
    max_steps: int


@dataclass
class SelfPlayResult:
    worker_id: int
    replay_buffer: list[dict]
    evaluator_statistics: EvaluatorStatistics


class SelfPlayManager(threading.Thread):
    def __init__(
        self, num_workers, base_random_seed, num_games, game_params: GameParams, alphazero_params: AlphaZeroParams
    ):
        self.num_workers = num_workers
        self.base_random_seed = base_random_seed
        self.num_games = num_games
        self.game_params = game_params
        self.alphazero_params = alphazero_params
        self._results = []
        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()
        super().__init__()

    def run(self):
        # Create the evaluator server and start its processing thread.
        action_encoder = ActionEncoder(self.game_params.board_size)
        nn_evaluator = NNEvaluator(action_encoder, my_device())

        # Queues used for worker processes to send evaluation requests to the EvaluatorServer, and for it
        # to send the resulting (value, policy) back.
        evaluator_request_queue = mp.Queue()
        evaluator_result_queues = [mp.Queue() for _ in range(self.num_workers)]

        # Create the worker processes. We hide the CUDA devices
        # in the worker processes since they're only doing CPU work,
        # and we don't want pytorch to waste GPU memory with the allocations
        # it automatically does on startup
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        games_per_worker = int(np.ceil(self.num_games / self.num_workers))
        games_remaining_to_allocate = self.num_games
        workers = []
        for worker_id in range(self.num_workers):
            this_worker_num_games = min(games_per_worker, games_remaining_to_allocate)

            evaluator_client = EvaluatorClient(
                self.game_params.board_size,
                worker_id,
                evaluator_request_queue,
                evaluator_result_queues[worker_id],
            )
            worker_process = mp.Process(
                target=run_self_play_games,
                args=(
                    this_worker_num_games,
                    self.game_params,
                    self.alphazero_params,
                    evaluator_client,
                    self.base_random_seed + worker_id,
                    worker_id,
                    self._result_queue,
                ),
            )
            workers.append(worker_process)
            worker_process.start()

        # Restore the CUDA_VISIBLE_DEVICES environment variable so that the server
        # can use the GPU if it wants to.
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        evaluator_server = EvaluatorServer(
            nn_evaluator,
            input_queue=evaluator_request_queue,
            output_queues=evaluator_result_queues,
        )
        evaluator_server.start()

        for worker in workers:
            worker.join()

        evaluator_server.shutdown()
        print(evaluator_server.get_statistics())
        evaluator_server.join()

    def get_results(self, timeout: float = 0.1) -> Optional[list[dict]]:
        """
        Get results if they are available, waiting up to timeout seconds.

        Args:
            timeout: Maximum time to wait for results in seconds.

        Returns:
            List of replay buffer dictionaries, or None on timeout.
        """
        t0 = time.time()
        while len(self._results) < self.num_workers:
            t1 = time.time()
            new_timeout = timeout - (t1 - t0)
            if new_timeout <= 0.0:
                break

            try:
                worker_result = self._result_queue.get(timeout=new_timeout)
            except mp.queues.Empty:
                break

            self._results.append(worker_result)
            if len(self._results) >= self.num_workers:
                # Merge results into one replay buffer. We sort them to make the results deterministic.
                replay_buffer = []
                results = sorted(self._results, key=lambda r: r.worker_id)
                for r in results:
                    print(r.evaluator_statistics)

                    # NOTE: Make sure the replay buffer size for the training agent is large enough to hold
                    # the replay buffer results from all agents each epoch or else we'll end up discarding
                    # some results and we'll have wasted computation by playing those games.
                    replay_buffer.extend(r.replay_buffer)

                return replay_buffer

        return None

    def shutdown(self):
        """
        Stop early and clean up any resources. No need to call this if run() has already returned.
        """
        self._stop_event.set()


def run_self_play_games(
    num_games: int,
    game_params: GameParams,
    alphazero_params: AlphaZeroParams,
    evaluator: EvaluatorClient,
    random_seed: int,
    worker_id: int,
    result_queue: mp.Queue,
):
    # Each worker process uses its own random seed to make sure they don't make the exact same moves during
    # their self-play moves.
    set_deterministic(random_seed)

    environment = quoridor_env.env(
        board_size=game_params.board_size,
        max_walls=game_params.max_walls,
        max_steps=game_params.max_steps,
        step_rewards=False,
    )

    # Use the updated model and parameters
    alphazero_agent = AlphaZeroAgent(
        game_params.board_size,
        game_params.max_walls,
        game_params.max_steps,
        params=alphazero_params,
        evaluator=evaluator,
    )

    for game_i in range(num_games):
        alphazero_agent.start_game(None, None)
        environment.reset()
        num_turns = 0

        for _ in environment.agent_iter():
            observation, _, termination, truncation, _ = environment.last()
            if termination:
                break
            elif truncation:
                print("Game was truncated")
                print(environment.render())
                break

            action_index = alphazero_agent.get_action(observation)
            environment.step(action_index)
            num_turns += 1

        print(f"Worker {worker_id}: Game {game_i + 1}/{num_games} ended after {num_turns} turns")
        alphazero_agent.end_game(environment)

    result_queue.put(SelfPlayResult(worker_id, alphazero_agent.replay_buffer, evaluator.get_statistics()))

    print(f"Worker {worker_id} exiting")
