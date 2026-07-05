import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from threading import Thread

import numpy as np
import wandb
from pydantic_yaml import parse_yaml_file_as
from utils import Timer
from v2.common import JobTrigger, MockWandb, ShutdownSignal, create_alphazero, upload_model
from v2.config import Config
from v2.yaml_models import GameInfo, LatestModel


class Sampler:
    def __init__(self, dir: Path, max_cached_games: int = 1000):
        self.dir = dir
        self.max_cached_games = max_cached_games
        self.cache: OrderedDict = OrderedDict()

    def remove_game(self, game_filename: str):
        self.cache.pop(game_filename, None)

    def _ensure_loaded(self, game_filename: str):
        if game_filename in self.cache:
            self.cache.move_to_end(game_filename)
        else:
            with np.load(self.dir / game_filename) as npz:
                self.cache[game_filename] = {
                    "input_arrays": npz["input_arrays"],
                    "policies": npz["policies"],
                    "action_masks": npz["action_masks"],
                    "values": npz["values"],
                    "players": npz["players"],
                }
            while len(self.cache) > self.max_cached_games:
                self.cache.popitem(last=False)

    def sample(self, game_filename: str, n: int):
        self._ensure_loaded(game_filename)
        data = self.cache[game_filename]
        indices = np.random.choice(data["values"].shape[0], n)
        return [
            {
                "input_array": data["input_arrays"][idx],
                "mcts_policy": data["policies"][idx],
                "action_mask": data["action_masks"][idx],
                "value": float(data["values"][idx]),
                "player": int(data["players"][idx]),
            }
            for idx in indices
        ]


def _should_skip_iteration(
    total_moves: int,
    batch_size: int,
    games_needed_to_train: int,
    last_game: int,
    selfplay_disabled: bool,
) -> bool:
    """Decide whether to skip this iteration of the trainer's main loop.

    Always skip when the buffer holds fewer moves than one batch. When self-play is
    enabled, also skip when the trainer is ahead of self-play (the
    `games_needed_to_train` gate). When self-play is disabled the buffer is static
    and there is no production cadence to wait on, so we train every iteration once
    enough moves are available.
    """
    if total_moves < batch_size:
        return True
    if selfplay_disabled:
        return False
    return games_needed_to_train > last_game


def _build_game_log(
    game_info,
    model_version: int,
    last_game: int,
    omit_model_lag: bool,
) -> dict:
    """Per-game wandb log payload emitted when a game is ingested from ready/.

    `model_lag` is meaningful only when games arrive from live self-play of the
    current run, since it compares the trainer's current model version against the
    version that *produced* the game. When games come from a preloaded buffer
    (different lineage), the subtraction is nonsense, so the caller asks us to
    omit the key.
    """
    log = {
        "game_length": game_info.game_length,
        "Game num": last_game,
        "Model version": model_version,
    }
    if not omit_model_lag:
        log["model_lag"] = model_version - 1 - game_info.model_version
    return log


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


def train(config: Config, games_already_trained_on: int = 0):
    """
    Main training loop.

    Args:
        config: Config object
        games_already_trained_on: used to determine when to train next. This is used when preloading a replay buffer from a
            previous run, so that the trainer doesn't train on the same games multiple times.
    """
    batch_size = config.training.batch_size
    selfplay_disabled = not config.self_play.enabled
    omit_model_lag = config.training.initial_replay_buffer is not None
    alphazero_agent = create_alphazero(config, config.self_play.alphazero, overrides={"training_mode": True})
    alphazero_agent.evaluator.setup_lr_scheduler(config.training.lr_scheduler)

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
    filename = config.paths.checkpoints / "model_0.pt"
    alphazero_agent.save_model(filename)

    if config.training.save_onnx:
        onnx_filename = config.paths.checkpoints / "model_0.onnx"
        alphazero_agent.save_model_onnx(onnx_filename)

    # Write latest.yaml after all model files are saved (including ONNX),
    # so that consumers (e.g. Rust self-play) don't try to load files that
    # haven't been written yet.
    LatestModel.write(config, str(filename), 0)

    finish_condition = None
    if config.training.finish_after:
        finish_condition = JobTrigger.from_string(config, config.training.finish_after)

    games_needed_to_train = games_already_trained_on + config.training.games_per_training_step
    last_game = 0
    total_moves_played = 0
    model_version = 1
    moves_per_game = []
    game_filename = []
    sampler = Sampler(config.paths.replay_buffers, config.training.max_cached_games)
    while True:
        if finish_condition and finish_condition.is_ready():
            print(f"Trainer: reached out finish condition: {config.training.finish_after}")
            break

        if ShutdownSignal.is_set(config):
            print("Shutdown file found.  Finishing training")
            break

        Timer.start("waiting-to-train", ignore_if_running=True)

        # Process new games: find new files, move them and extract the info used for training
        ready = [f for f in sorted(config.paths.replay_buffers_ready.glob("*.npz")) if f.is_file()]

        for f in ready:
            last_game += 1

            new_name = config.paths.replay_buffers / f"game_{last_game:07d}.npz"

            yaml_file = f.with_suffix(".yaml")
            new_yaml_name = new_name.with_suffix(".yaml")
            yaml_file.rename(new_yaml_name)
            game_info = parse_yaml_file_as(GameInfo, new_yaml_name)

            f.rename(new_name)
            moves_per_game.append(game_info.game_length)
            total_moves_played += game_info.game_length
            game_filename.append(new_name.name)
            wandb_run.log(_build_game_log(game_info, model_version, last_game, omit_model_lag))

        # Trim oldest games to stay within the replay buffer size limit
        while len(moves_per_game) > config.training.replay_buffer_size:
            moves_per_game.pop(0)
            f = game_filename.pop(0)
            sampler.remove_game(f)

        total_moves = sum(moves_per_game)

        if _should_skip_iteration(
            total_moves=total_moves,
            batch_size=batch_size,
            games_needed_to_train=games_needed_to_train,
            last_game=last_game,
            selfplay_disabled=selfplay_disabled,
        ):
            time.sleep(1)
            continue

        time_waiting_to_train = Timer.finish("waiting-to-train")

        # Sample moves from the replay buffer files
        Timer.start("sample")
        samples = []

        buffer_size = len(moves_per_game)
        games = np.random.choice(buffer_size, batch_size, p=[moves / total_moves for moves in moves_per_game])
        samples_per_game = Counter(games)
        for game_number in samples_per_game:
            samples.extend(sampler.sample(game_filename[game_number], samples_per_game[game_number]))

        time_sample = Timer.finish("sample")

        # Train the network for one step using the samples
        Timer.start("train")
        policy_loss, value_loss, total_loss = alphazero_agent.evaluator.train_iteration_v2(samples)
        games_needed_to_train += config.training.games_per_training_step
        time_train = Timer.finish("train")

        wandb_run.log(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
                "learning_rate": alphazero_agent.evaluator.get_learning_rate(),
                "games_played": last_game,
                "moves_played": total_moves_played,
                "replay_buffer_games": buffer_size,
                "replay_buffer_moves": total_moves,
                "time-sample": time_sample,
                "time-train": time_train,
                "time-waiting-to-train": time_waiting_to_train,
                "Model version": model_version,
            },
            commit=True,
        )

        Timer.start("save-model")

        # Save in PyTorch format
        new_model_filename = config.paths.checkpoints / f"model_{model_version}.pt"
        alphazero_agent.save_model(new_model_filename)

        # Save in ONNX format if enabled
        if config.training.save_onnx:
            onnx_model_filename = config.paths.checkpoints / f"model_{model_version}.onnx"
            alphazero_agent.save_model_onnx(onnx_model_filename)

        # Write latest.yaml after all model files are saved
        LatestModel.write(config, str(new_model_filename), model_version)

        time_save_model = Timer.finish("save-model")

        if config.training.model_save_timing:
            formats = []
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
