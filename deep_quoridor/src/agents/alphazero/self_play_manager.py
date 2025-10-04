import hashlib
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from queue import Empty
from typing import Optional

import quoridor_env
from utils import set_deterministic

from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from agents.alphazero.nn_evaluator import EvaluatorStatistics


@dataclass
class GameParams:
    board_size: int
    max_walls: int
    max_steps: int


@dataclass
class SelfPlayResult:
    worker_id: int
    replay_buffer: list[dict]
    evaluator_statistics: Optional[EvaluatorStatistics]


class SelfPlayManager(threading.Thread):
    def __init__(
        self,
        num_workers,
        base_random_seed,
        epoch: int,
        num_games,
        game_params: GameParams,
        alphazero_params: AlphaZeroParams,
        num_parallel_games: int,
    ):
        self.num_workers = num_workers
        self.base_random_seed = base_random_seed
        self.epoch = epoch
        self.num_games = num_games
        self.game_params = game_params
        self.alphazero_params = alphazero_params
        self._results = []
        self._result_queue = mp.Queue()
        self._stop_event = mp.Event()
        self.num_parallel_games = num_parallel_games
        super().__init__()

    # Return an array with the number of games per worker, such that they add up to the total number of games
    # and each worker gets X or X+1 jobs.
    # E.g. for 20 games and 8 workers it returns: [3, 3, 3, 3, 2, 2, 2, 2]
    def _games_per_worker(self):
        base = self.num_games // self.num_workers
        extra = self.num_games - base * self.num_workers
        return [base + 1 if i < extra else base for i in range(self.num_workers)]

    def run(self):
        games_per_worker = self._games_per_worker()
        workers = []
        for worker_id in range(self.num_workers):
            worker_process = mp.Process(
                target=run_self_play_games,
                args=(
                    games_per_worker[worker_id],
                    self.game_params,
                    self.alphazero_params,
                    self.compute_worker_random_seed(worker_id),
                    worker_id,
                    self._result_queue,
                    self.num_parallel_games,
                ),
            )
            workers.append(worker_process)
            worker_process.start()

        for worker in workers:
            worker.join()

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
            except Empty:
                break

            self._results.append(worker_result)
            if len(self._results) >= self.num_workers:
                # Merge results into one replay buffer. We sort them to make the results deterministic.
                replay_buffer = []
                results = sorted(self._results, key=lambda r: r.worker_id)
                for r in results:
                    print(f"Worker {r.worker_id} replay buffer size: {len(r.replay_buffer)}")
                    print(r.evaluator_statistics)

                    # NOTE: Make sure the replay buffer size for the training agent is large enough to hold
                    # the replay buffer results from all agents each epoch or else we'll end up discarding
                    # some results and we'll have wasted computation by playing those games.
                    replay_buffer.extend(r.replay_buffer)

                return replay_buffer

        return None

    def compute_worker_random_seed(self, worker_id: int) -> int:
        """
        This feels like overkill but we want a random seed for each worker that is:

            - different from all other workers
            - different each epoch
            - different from the base seed used by the main process
            - the same if we re-run the script with the same arguments and same random seed
            - fits in an int32 (numpy, for example, requires this for its random seed)
        """

        worker_random_hash = hashlib.sha256()
        for x in (self.base_random_seed, worker_id, self.epoch):
            worker_random_hash.update(x.to_bytes((x.bit_length() + 7) // 8, byteorder="big"))
        worker_random_seed = int.from_bytes(worker_random_hash.digest()[:4], byteorder="big")
        return worker_random_seed

    def shutdown(self):
        """
        Stop early and clean up any resources. No need to call this if run() has already returned.
        """
        self._stop_event.set()


def run_self_play_games(
    num_games: int,
    game_params: GameParams,
    alphazero_params: AlphaZeroParams,
    random_seed: int,
    worker_id: int,
    result_queue: mp.Queue,
    num_parallel_games: int,
):
    # Each worker process uses its own random seed to make sure they don't make the exact same moves during
    # their self-play moves.
    set_deterministic(random_seed)

    print(
        f"Worker {worker_id} starting, running {num_games} games ({num_parallel_games} in parallel) with random seed {random_seed}"
    )

    environments = [
        quoridor_env.env(
            board_size=game_params.board_size,
            max_walls=game_params.max_walls,
            max_steps=game_params.max_steps,
            step_rewards=False,
        )
        for i in range(num_parallel_games)
    ]

    # Use the updated model and parameters
    alphazero_agent = AlphaZeroAgent(
        game_params.board_size,
        game_params.max_walls,
        game_params.max_steps,
        params=alphazero_params,
    )

    game_i = 0
    while game_i < num_games:
        n = min(num_parallel_games, num_games - game_i)
        environments = environments[:n]  # mhhh
        for i in range(n):
            environments[i].reset()

        # TODO problems with state
        # alphazero_agent.start_game(None, None)
        alphazero_agent.multi_start_game(environments)

        # print(f"=== Playing {n} games in parallel, {len(environments)} envs")

        num_turns = [0] * n
        finished = [False] * n

        t0 = time.time()
        while not all(finished):
            observations = []
            for i in range(n):
                if finished[i]:
                    continue

                # print(f"game {i}, turn {num_turns[i]}")

                observation, _, termination, truncation, _ = environments[i].last()
                if termination:
                    print(f"Worker {worker_id} :Game {game_i + i + 1}/{num_games} ended after {num_turns[i]} turns")
                    finished[i] = True
                elif truncation:
                    print(f"Worker {worker_id}: Game {game_i + i + 1}/{num_games} truncated after {num_turns[i]} turns")
                    print(environments[i].render())
                    finished[i] = True
                else:
                    observations.append((i, observation))
                    num_turns[i] += 1

            action_indexes = alphazero_agent.multi_get_action(observations)
            # print(f"action indexes: {action_indexes}")
            # print(f"finished: {finished}, obs: {len(observations)}, turns {num_turns}")
            for env_idx, action_index in action_indexes:
                if finished[env_idx]:
                    continue
                # print(f"game {env_idx}, action {action_index}")
                environments[env_idx].step(action_index)

        game_i += n
        t1 = time.time()
        alphazero_agent.multi_end_game()
        # for env in environments:
        #     alphazero_agent.end_game(env)

        print(f"Worker {worker_id}: played {n} games in ({t1 - t0:.2f}s)")

    # TODO implement the stats for the per process evaluator
    result_queue.put(
        SelfPlayResult(
            worker_id,
            list(alphazero_agent.replay_buffer),
            None,
        )
    )

    print(f"Worker {worker_id} exiting")
