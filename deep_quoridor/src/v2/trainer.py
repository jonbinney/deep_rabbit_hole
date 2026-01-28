import pickle
import time
from collections import Counter

import numpy as np
import wandb
from v2.common import MockWandb, create_alphazero
from v2.config import Config
from v2.yaml_models import LatestModel


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
    game_filename = []

    while True:
        while True:
            ready = [f for f in sorted(config.paths.replay_buffers_ready.glob("*.pkl")) if f.is_file()]
            if len(ready) >= min_new_games:
                break
            time.sleep(1)

        # Process new games
        for f in ready:
            last_game += 1

            new_name = config.paths.replay_buffers / f"game_{last_game:07d}.pkl"

            yaml_file = f.with_suffix(".yaml")
            new_yaml_name = new_name.with_suffix(".yaml")
            yaml_file.rename(new_yaml_name)

            f.rename(new_name)
            with open(new_name, "rb") as f:
                data = pickle.load(f)
                game_length = len(list(data))
                moves_per_game.append(game_length)
                game_filename.append(f.name)
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
                file = config.paths.replay_buffers / game_filename[game_number]
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
