import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import wandb
from config import Config, load_config_and_setup_run
from v2 import LatestModel, benchmarks, create_alphazero
from v2.common import MockWandb

# TO DO
sys.path.insert(0, str(Path(__file__).parent.parent))


import quoridor_env
from agents.alphazero.alphazero import AlphaZeroAgent, AlphaZeroParams
from utils.subargs import parse_subargs

azparams_str = """training_mode=true,nn_type=resnet,nn_resnet_num_blocks=2,mcts_ucb_c=1.2,\
mcts_noise_epsilon=0.25,mcts_n=500,\
learning_rate=0.001,batch_size=2048,optimizer_iterations=1"""

azparams = parse_subargs(azparams_str, AlphaZeroParams)


def self_play(config: Config):
    LatestModel.wait_for_creation(config)
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

    # run_id = f"mock-self-play-{os.getpid()}"
    # wandb_run = wandb.init(
    #     project="v2",
    #     job_type="self-play",
    #     group=run_name,
    #     name=run_id,
    #     id=run_id,
    #     resume="allow",
    # )
    # Timer.set_wandb_run(wandb_run)

    while True:
        latest = LatestModel.load(config)

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
        filenames = alphazero_agent.end_game_batch_and_save_replay_buffers(config.paths.replay_buffers_tmp)
        for i, f in enumerate(filenames):
            # TODO maybe pad with 0 the version
            os.rename(
                f,
                config.paths.replay_buffers_ready
                / f"game_model_{latest.version}_{os.getpid()}_{int(t0 * 1000)}_{i}.pkl",
            )

        num_truncated = n - len(finished_in)

        print(f"{os.getpid()} - finsihed in {t1 - t0} {sorted(finished_in)}, {num_truncated}")
        # print(
        #     f"Worker {worker_id}: ({t1 - t0:.2f}s) Games {game_i}...{game_i + n - 1} / {num_games} ended after {sorted(finished_in)} turns. {num_truncated} truncated"
        # )


def train(config: Config):
    global azparams
    batch_size = config.training.batch_size
    training_iterations = 1
    min_new_games = 25

    if config.wandb:
        run_id = f"{config.run_id}-training"
        wandb_run = wandb.init(
            project=config.wandb.project,
            job_type="training",
            group=config.run_id,
            name=run_id,
            id=run_id,
            resume="allow",
        )
        wandb.define_metric("Game num", hidden=True)
        wandb.define_metric("Model version", hidden=True)
        wandb.define_metric("game_length", "Game num")
        wandb.define_metric("*", "Model version")
    else:
        wandb_run = MockWandb()

    alphazero_agent = create_alphazero(config, config.self_play.alphazero, overrides={"training_mode": True})

    filename = config.paths.checkpoints / "model_0.pt"
    alphazero_agent.save_model(filename)
    LatestModel.write(config, str(filename), 0)

    last_game = 0
    model_version = 1
    moves_per_game = []
    while True:
        while True:
            ready = [f for f in sorted(config.paths.replay_buffers_ready.iterdir()) if f.is_file()]
            if len(ready) >= min_new_games:
                break
            time.sleep(1)

        # Process new games
        for f in ready:
            last_game += 1
            # TO DO: include model version in the name
            # TO DO: send it to a dir based on % 1000
            new_name = config.paths.replay_buffers / f"game_{last_game}.pkl"
            os.rename(f, new_name)
            with open(new_name, "rb") as f:
                data = pickle.load(f)
                game_length = len(list(data))
                moves_per_game.append(game_length)
                wandb_run.log({"game_length": game_length, "Game num": last_game, "Model version": model_version})

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
                file = config.paths.replay_buffers / f"game_{game_number + 1}.pkl"
                with open(file, "rb") as f:
                    data = pickle.load(f)

                samples.extend(np.random.choice(list(data), samples_per_game[game_number]))

                # print(f"{game_number}: {samples_per_game[game_number]}, {len(entries)}")

            # Train
            loss = alphazero_agent.evaluator.train_iteration_v2(samples)
            wandb_run.log({"loss": loss, "games_played": last_game, "Model version": model_version}, commit=True)

        print(f"Loss: {loss}")
        t1 = time.time()
        print(f"Sampling and training took {t1 - t0}")

        new_model_filename = config.paths.checkpoints / f"model_{model_version}.pt"
        alphazero_agent.save_model(new_model_filename)
        LatestModel.write(config, str(new_model_filename), model_version)
        model_version += 1


if __name__ == "__main__":
    config = load_config_and_setup_run(
        "deep_quoridor/experiments/B5W3/base.yaml", "/Users/amarcu/code/deep_rabbit_hole"
    )
    mp.set_start_method("spawn", force=True)

    processes = []

    p = mp.Process(target=train, args=[config])
    p.start()
    processes.append(p)

    ps = benchmarks.create_benchmark_processes(config)
    [p.start() for p in ps]
    processes.extend(ps)

    num_workers = 2
    for i in range(num_workers):
        p = mp.Process(target=self_play, args=[config])
        p.start()
        processes.append(p)

    for worker in processes:
        worker.join()
