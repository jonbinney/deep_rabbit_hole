import pickle
import threading
import time
from collections import Counter
from threading import Thread

import numpy as np
import wandb
from pydantic_yaml import parse_yaml_file_as
from utils import Timer
from v2.common import JobTrigger, MockWandb, ShutdownSignal, create_alphazero, upload_model
from v2.config import Config
from v2.yaml_models import GameInfo, LatestModel


def model_uploader(config: Config, every: str, model_id: str, wandb_run, shutdown_event: threading.Event):
    LatestModel.wait_for_creation(config)

    trigger = JobTrigger.from_string(config, every)
    while True:
        latest = LatestModel.load(config)
        aliases = [f"m{latest.version}-{config.run_id}"]
        upload_model(wandb_run, config, latest, model_id, aliases)

        if shutdown_event.is_set():
            return

        # wait until the next time that we need to upload a model or for the shutdown signal.
        # If we get the shutdown signal, we'll do 1 more loop of the while to upload the last model.
        trigger.wait(lambda: shutdown_event.is_set())


def train(config: Config):
    batch_size = config.training.batch_size

    alphazero_agent = create_alphazero(config, config.self_play.alphazero, overrides={"training_mode": True})

    upload_model_thread = None
    shutdown_event = None
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

        if config.wandb.upload_model and config.wandb.upload_model.every:
            shutdown_event = threading.Event()
            upload_model_thread = Thread(
                target=model_uploader,
                args=(config, config.wandb.upload_model.every, alphazero_agent.model_id(), wandb_run, shutdown_event),
            )
            upload_model_thread.start()

    else:
        wandb_run = MockWandb()

    # Save initial model (model_0)
    if config.training.save_pytorch:
        filename = config.paths.checkpoints / "model_0.pt"
        alphazero_agent.save_model(filename)
        LatestModel.write(config, str(filename), 0)
    
    if config.training.save_onnx:
        onnx_filename = config.paths.checkpoints / "model_0.onnx"
        alphazero_agent.save_model_onnx(onnx_filename)

    finish_condition = None
    if config.training.finish_after:
        finish_condition = JobTrigger.from_string(config, config.training.finish_after)

    training_steps = 0
    last_game = 0
    model_version = 1
    moves_per_game = []
    game_filename = []

    while True:
        if finish_condition and finish_condition.is_ready():
            print(f"Trainer: reached out finish condition: {config.training.finish_after}")
            break

        if ShutdownSignal.is_set(config):
            print("Shutdown file found.  Finishing training")
            break

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

        Timer.start("save-model")
        
        # Save in PyTorch format if enabled
        if config.training.save_pytorch:
            new_model_filename = config.paths.checkpoints / f"model_{model_version}.pt"
            alphazero_agent.save_model(new_model_filename)
            LatestModel.write(config, str(new_model_filename), model_version)
        
        # Save in ONNX format if enabled
        if config.training.save_onnx:
            onnx_model_filename = config.paths.checkpoints / f"model_{model_version}.onnx"
            alphazero_agent.save_model_onnx(onnx_model_filename)
        
        time_save_model = Timer.finish("save-model")
        
        if config.training.model_save_timing:
            formats = []
            if config.training.save_pytorch:
                formats.append("PyTorch")
            if config.training.save_onnx:
                formats.append("ONNX")
            format_str = " and ".join(formats) if formats else "no format"
            print(f"Saving model ({format_str}) took {time_save_model:.4f}s")
        
        model_version += 1

    ShutdownSignal.signal(config)
    if upload_model_thread and shutdown_event:
        shutdown_event.set()
        upload_model_thread.join()
