import pickle
import time
from collections import Counter
from threading import Thread

import numpy as np
import wandb
from pydantic_yaml import parse_yaml_file_as
from utils import Timer
from v2.common import JobTrigger, MockWandb, create_alphazero
from v2.config import Config
from v2.yaml_models import GameInfo, LatestModel


def model_uploader(config: Config):
    LatestModel.wait_for_creation(config)

    every = config.wandb.upload_model.every
    if not every:
        return
    trigger = JobTrigger.from_string(config, every)
    while True:
        trigger.wait()
        print(f"Time to upload model! {time.time()}")
        latest_model_filename = LatestModel.load(config).filename
        artifact = wandb.Artifact("alphazero_B5W3_mv1.0", type="model")
        artifact.add_file(local_path=latest_model_filename)
        artifact.save()

        print(f"Uploaded! {time.time}")


def train(config: Config):
    batch_size = config.training.batch_size

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

        if config.wandb.upload_model:
            upload_model_thread = Thread(target=model_uploader, args=(config,))
            upload_model_thread.start()

    else:
        wandb_run = MockWandb()

    alphazero_agent = create_alphazero(config, config.self_play.alphazero, overrides={"training_mode": True})

    filename = config.paths.checkpoints / "model_0.pt"
    alphazero_agent.save_model(filename)
    LatestModel.write(config, str(filename), 0)

    training_steps = 0
    last_game = 0
    model_version = 1
    moves_per_game = []
    game_filename = []

    while True:
        Timer.start("waiting-to-train", ignore_if_running=True)

        # Process new games: find new files, move them and extract the info used for training
        ready = [f for f in sorted(config.paths.replay_buffers_ready.glob("*.pkl")) if f.is_file()]

        for f in ready:
            last_game += 1

            new_name = config.paths.replay_buffers / f"game_{last_game:07d}.pkl"

            yaml_file = f.with_suffix(".yaml")
            new_yaml_name = new_name.with_suffix(".yaml")
            yaml_file.rename(new_yaml_name)
            game_info = parse_yaml_file_as(GameInfo, new_yaml_name)

            f.rename(new_name)
            with open(new_name, "rb") as f:
                data = pickle.load(f)
                moves_per_game.append(game_info.game_length)
                game_filename.append(f.name)
                wandb_run.log(
                    {
                        "game_length": game_info.game_length,
                        "model_lag": model_version - 1 - game_info.model_version,
                        "Game num": last_game,
                        "Model version": model_version,
                    }
                )

        total_moves = sum(moves_per_game)

        games_needed_to_train = config.training.games_per_training_step * (training_steps + 1)

        if total_moves < batch_size or games_needed_to_train > last_game:
            time.sleep(1)
            continue

        time_waiting_to_train = Timer.finish("waiting-to-train")

        # Sample moves from the replay buffer files
        Timer.start("sample")
        samples = []

        games = np.random.choice(last_game, batch_size, p=[moves / total_moves for moves in moves_per_game])
        samples_per_game = Counter(games)
        for game_number in samples_per_game:
            file = config.paths.replay_buffers / game_filename[game_number]
            with open(file, "rb") as f:
                data = pickle.load(f)

            samples.extend(np.random.choice(list(data), samples_per_game[game_number]))
        time_sample = Timer.finish("sample")

        # Train the network for one step using the samples
        Timer.start("train")
        policy_loss, value_loss, total_loss = alphazero_agent.evaluator.train_iteration_v2(samples)
        training_steps += 1
        time_train = Timer.finish("train")

        wandb_run.log(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
                "games_played": last_game,
                "time-sample": time_sample,
                "time-train": time_train,
                "time-waiting-to-train": time_waiting_to_train,
                "Model version": model_version,
            },
            commit=True,
        )

        print(f"Sampling and training took {time_sample}, {time_train}")

        new_model_filename = config.paths.checkpoints / f"model_{model_version}.pt"
        alphazero_agent.save_model(new_model_filename)
        LatestModel.write(config, str(new_model_filename), model_version)
        model_version += 1

    # TODO shutdown upload_model_thread
