import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import wandb
from pydantic_yaml import parse_yaml_file_as, to_yaml_file

# TO DO
sys.path.insert(0, str(Path(__file__).parent.parent))


import quoridor_env
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from metrics import Metrics
from pydantic import BaseModel
from utils import Timer
from utils.subargs import override_subargs, parse_subargs

azparams_str = """training_mode=true,nn_type=resnet,nn_resnet_num_blocks=2,mcts_ucb_c=1.2,\
mcts_noise_epsilon=0.25,mcts_n=500,\
learning_rate=0.001,batch_size=2048,optimizer_iterations=1"""

azparams = parse_subargs(azparams_str, AlphaZeroParams)

run_name = "mock7"

run_dir = Path("/Users/amarcu/code/deep_rabbit_hole/runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)  # just for convenience

models_dir = run_dir / "models"

checkpoints_dir = models_dir / "checkpoints"
checkpoints_dir.mkdir(parents=True, exist_ok=True)

replay_buffers_dir = run_dir / "replay_buffers"
ready_replay_buffers_dir = replay_buffers_dir / "ready"
tmp_replay_buffers_dir = ready_replay_buffers_dir / "tmp"
tmp_replay_buffers_dir.mkdir(parents=True, exist_ok=True)

logs_dir = run_dir / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)


class LatestModel(BaseModel):
    filename: str
    version: int


def self_play():
    n = 8
    environments = [
        quoridor_env.env(
            board_size=5,
            max_walls=3,
            max_steps=30,
        )
        for _ in range(n)
    ]
    global azparams
    alphazero_agent = AlphaZeroAgent(
        5,
        3,
        30,
        params=azparams,
    )
    current_model_version = -1

    run_id = f"mock-self-play-{os.getpid()}"
    wandb_run = wandb.init(
        project="v2",
        job_type="self-play",
        group=run_name,
        name=run_id,
        id=run_id,
        resume="allow",
    )
    Timer.set_wandb_run(wandb_run)

    while True:
        latest = parse_yaml_file_as(LatestModel, models_dir / "latest.yaml")

        if latest.version != current_model_version:
            print(f"Loading model version {latest.version} - {os.getpid()}")
            current_model_version = latest.version
            alphazero_agent.load_model(latest.filename)

        for e in environments:
            e.reset()

        alphazero_agent.start_game_batch(environments)

        num_turns = [0] * n
        finished_in = []
        finished = [False] * n

        t0 = time.time()
        while not all(finished):
            observations = []
            for i in range(n):
                if finished[i]:
                    continue

                observation, _, termination, truncation, _ = environments[i].last()
                if termination:
                    finished[i] = True
                    finished_in.append(num_turns[i])
                elif truncation:
                    finished[i] = True
                else:
                    observations.append((i, observation))
                    num_turns[i] += 1

            action_index_batch = alphazero_agent.get_action_batch(observations)

            for env_idx, action_index in action_index_batch:
                if finished[env_idx]:
                    continue
                environments[env_idx].step(action_index)
                # game_actions[env_idx + game_i].append(action_index)

        t1 = time.time()
        filenames = alphazero_agent.end_game_batch_and_save_replay_buffers(tmp_replay_buffers_dir)
        for i, f in enumerate(filenames):
            # TODO maybe pad with 0 the version
            os.rename(
                f, ready_replay_buffers_dir / f"game_model_{latest.version}_{os.getpid()}_{int(t0 * 1000)}_{i}.pkl"
            )

        num_truncated = n - len(finished_in)

        print(f"{os.getpid()} - finsihed in {t1 - t0} {sorted(finished_in)}, {num_truncated}")
        # print(
        #     f"Worker {worker_id}: ({t1 - t0:.2f}s) Games {game_i}...{game_i + n - 1} / {num_games} ended after {sorted(finished_in)} turns. {num_truncated} truncated"
        # )


def train():
    global azparams
    batch_size = 2048
    training_iterations = 1
    min_new_games = 25

    run_id = f"mock-trainer-{os.getpid()}"
    wandb_run = wandb.init(
        project="v2",
        job_type="trainer",
        group=run_name,
        name=run_id,
        id=run_id,
        resume="allow",
    )

    alphazero_agent = AlphaZeroAgent(
        5,
        3,
        30,
        params=azparams,
    )
    filename = checkpoints_dir / "model_0.pt"
    alphazero_agent.save_model(filename)
    latest = LatestModel(
        filename=str(filename),
        version=0,
    )

    to_yaml_file(models_dir / "latest.yaml", latest)

    last_game = 0
    model_version = 1
    moves_per_game = []
    while True:
        while True:
            ready = [f for f in sorted(ready_replay_buffers_dir.iterdir()) if f.is_file()]
            if len(ready) >= min_new_games:
                break
            time.sleep(1)

        # Process new games
        for f in ready:
            last_game += 1
            # TO DO: include model version in the name
            # TO DO: send it to a dir based on % 1000
            new_name = replay_buffers_dir / f"game_{last_game}.pkl"
            os.rename(f, new_name)
            with open(new_name, "rb") as f:
                data = pickle.load(f)
                moves_per_game.append(len(list(data)))

        total_moves = sum(moves_per_game)
        if total_moves < batch_size:
            continue

        t0 = time.time()
        for _ in range(training_iterations):
            # Sample
            # TO DO, we need to roll out games when it's longer that the replay buffer size
            # TO DO probably we want to sample for all the training iterations together to make it faster
            samples = []

            games = np.random.choice(last_game, batch_size, p=[moves / total_moves for moves in moves_per_game])
            samples_per_game = Counter(games)
            for game_number in samples_per_game:
                file = replay_buffers_dir / f"game_{game_number + 1}.pkl"
                with open(file, "rb") as f:
                    data = pickle.load(f)

                samples.extend(np.random.choice(list(data), samples_per_game[game_number]))

                # print(f"{game_number}: {samples_per_game[game_number]}, {len(entries)}")

            # Train
            loss = alphazero_agent.evaluator.train_iteration_v2(samples)
            wandb_run.log({"loss": loss, "games_played": last_game}, step=model_version + 1, commit=True)

        print(f"Loss: {loss}")
        t1 = time.time()
        print(f"Sampling and training took {t1 - t0}")

        new_model_filename = checkpoints_dir / f"model_{model_version}.pt"
        alphazero_agent.save_model(new_model_filename)
        latest = LatestModel(
            filename=str(new_model_filename),
            version=model_version,
        )
        model_version += 1

        to_yaml_file(models_dir / "latest.yaml", latest)


def benchmarks():
    latest_yaml_path = models_dir / "latest.yaml"
    time.sleep(10)  # to do, just wait until it's available or a trigger
    run_id = f"mock-benchmark-{os.getpid()}"
    wandb_run = wandb.init(
        project="v2",
        job_type="self_play_worker",
        group=run_name,
        name=run_id,
        id=run_id,
        resume="allow",
    )
    while True:
        latest = parse_yaml_file_as(LatestModel, latest_yaml_path)

        az_params_override = override_subargs(
            azparams_str, {"mcts_n": 0, "model_filename": latest.filename, "training_mode": False}
        )
        params = f"alphazero:{az_params_override}"

        players: list[str] = [
            "greedy",
            "greedy:p_random=0.1,nick=greedy-01",
            "greedy:p_random=0.3,nick=greedy-03",
            "random",
            "simple:branching_factor=8,nick=simple-bf8",
            "simple:branching_factor=16,nick=simple-bf16",
        ]
        m = Metrics(5, 3, players, 10, 30, 1)
        print("METRICS - starting computation")
        (
            _,
            _,
            relative_elo,
            win_perc,
            p1_stats,
            p2_stats,
            absolute_elo,
            dumb_score,
        ) = m.compute(params)

        print(f"METRICS {latest.version}: {relative_elo=}, {win_perc=}, {dumb_score=}")
        wandb_run.log(
            {"win_perc": win_perc, "relative_elo": relative_elo, "dumb_score": dumb_score},
            step=latest.version,
            commit=True,
        )
        time.sleep(60)


if __name__ == "__main__":
    processes = []

    p = mp.Process(target=train)
    p.start()
    processes.append(p)

    p = mp.Process(target=benchmarks)
    p.start()
    processes.append(p)

    # Waiting for latest.yaml to exist
    timeout = 60  # seconds
    start_time = time.time()
    latest_yaml_path = models_dir / "latest.yaml"
    while not latest_yaml_path.exists():
        if time.time() - start_time > timeout:
            raise RuntimeError(f"Timeout: {latest_yaml_path} not found after {timeout} seconds.")
        time.sleep(1)

    num_workers = 2
    for i in range(num_workers):
        p = mp.Process(target=self_play)
        p.start()
        processes.append(p)

    for worker in processes:
        worker.join()
