import os

import quoridor_env
from utils import Timer
from v2.common import ShutdownSignal, create_alphazero
from v2.config import Config
from v2.yaml_models import LatestModel


def self_play(config: Config):
    LatestModel.wait_for_creation(config)
    n = config.self_play.parallel_games

    environments = [
        quoridor_env.env(
            board_size=config.quoridor.board_size,
            max_walls=config.quoridor.max_walls,
            max_steps=config.quoridor.max_steps,
        )
        for _ in range(n)
    ]

    agent = create_alphazero(config, config.self_play.alphazero, overrides={"training_mode": True})

    current_model_version = -1

    while not ShutdownSignal.is_set(config):
        latest = LatestModel.load(config)

        if latest.version != current_model_version:
            print(f"Loading model version {latest.version} - {os.getpid()}")
            current_model_version = latest.version
            agent.load_model(latest.filename)

        agent.start_game_batch(environments)

        num_turns = [0] * n
        finished_in = []
        finished = [False] * n

        Timer.start("self-play")
        while not all(finished):
            if ShutdownSignal.is_set(config):
                print(f"self-play process {os.getpid()} - Interrupted by shutdown signal")
                return

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

            action_index_batch = agent.get_action_batch(observations)

            for env_idx, action_index in action_index_batch:
                if finished[env_idx]:
                    continue
                environments[env_idx].step(action_index)

        agent.end_game_batch_and_save_replay_buffers(
            config.paths.replay_buffers_tmp, config.paths.replay_buffers_ready, current_model_version
        )
        num_truncated = n - len(finished_in)
        elapsed = Timer.finish("self-play")
        print(f"{os.getpid()} - finsihed in {elapsed} {sorted(finished_in)}, {num_truncated}")
