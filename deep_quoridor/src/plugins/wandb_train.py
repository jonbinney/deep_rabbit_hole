import datetime
import getpass
import os
from dataclasses import asdict, dataclass
from typing import List

import wandb
from agents import Agent
from agents.core.trainable_agent import TrainableAgent
from arena_utils import ArenaPlugin
from metrics import Metrics
from utils import resolve_path
from utils.subargs import SubargsBase


@dataclass
class WandbParams(SubargsBase):
    # Prefix for this run. This is used to create a unique run id for naming, and tagging artifacts and files
    prefix: str = getpass.getuser()

    # Suffix for this run. This is used to create a unique run id for naming, and tagging artifacts and files
    suffix: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # Name of the project in wandb
    project: str = "deep_quoridor"

    # Optional notes to store for the run
    notes: str = ""

    # Wether to upload the final model to wandb
    upload_model: bool = True

    # How often to log training metrics
    log_every: int = 10

    def run_id(self):
        return f"{self.prefix}-{self.suffix}"


class WandbTrainPlugin(ArenaPlugin):
    def __init__(
        self, params: WandbParams, total_episodes: int, agent_encoded_name: str, benchmarks: List[str | Agent] = []
    ):
        self.params = params
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.agent = None
        self.agent_encoded_name = agent_encoded_name
        self.best_model_filename = None
        # Notice that the best model won't be uploaded if it's not better than the initialization.
        self.best_model_relative_elo = -800
        self.benchmarks = benchmarks

    def _initialize(self, game):
        assert self.agent

        config = {
            "board_size": game.board_size,
            "max_walls": game.max_walls,
            "episodes": self.total_episodes,
            "player_args": self.agent.params,
        }
        config.update(self.agent.model_hyperparameters())
        self.metrics = Metrics(game.board_size, game.max_walls, self.benchmarks)

        self.run = wandb.init(
            project=self.params.project,
            job_type="train",
            config=config,
            tags=[self.agent.model_id(), f"-{self.params.run_id()}"],
            id=self.params.run_id(),
            name=f"{self.params.run_id()}",
            notes=self.params.notes,
        )

    def start_game(self, game, agent1, agent2):
        if (self.agent is not None) and (self.agent != agent1) and (self.agent != agent2):
            raise ValueError("WandbTrainPlugin being used for an agent, but another agent is being trained")
        if self.agent is not None:
            return
        if isinstance(agent1, TrainableAgent) and agent1.is_training():
            self.agent = agent1
        elif isinstance(agent2, TrainableAgent) and agent2.is_training():
            self.agent = agent2
        else:
            raise ValueError("WandbTrainPlugin can only be used with a training agent, both agents are not training")
        self._initialize(game)

    def end_game(self, game, result):
        assert self.agent
        self.episode_count += 1
        if self.episode_count % self.params.log_every == 0 or self.episode_count == self.total_episodes:
            avg_loss, avg_reward = self.agent.compute_loss_and_reward(self.params.log_every)

            self.run.log(
                {"loss": avg_loss, "reward": avg_reward, "epsilon": self.agent.epsilon},
                step=self.episode_count,
            )

    def _upload_model(self, save_file: str, aliases: list[str] | None = None) -> str:
        assert self.agent

        artifact = wandb.Artifact(f"{self.agent.model_id()}", type="model", metadata=asdict(self.agent.params))
        artifact.add_file(local_path=save_file)
        artifact.save()
        wandb.log_artifact(artifact, aliases=aliases).wait(60)
        print(f"Done! Model uploaded with version {artifact.version} and aliases {artifact.aliases}")

        wand_file = resolve_path(self.agent.params.wandb_dir, self.agent.wandb_local_filename(artifact))

        # Now that we know the digest, rename the file to include it, so it takes the expected name and
        # doesn't need to be re-downloaded from wandb.
        # Source and target file are in the same path, just a different file name
        os.rename(save_file, wand_file)
        print(f"Model saved to {wand_file}")
        return str(wand_file)

    def end_arena(self, game, results):
        assert self.agent
        if not self.params.upload_model:
            print("Model NOT uploaded to wandb since using `upload_model=False`")
            wandb.finish()
            return

        # Save the model in the wandb directory with the suffix "final".  The file will be renamed
        # once we upload it to wandb and have the digest.
        save_file = resolve_path(self.agent.params.wandb_dir, self.agent.resolve_filename("final"))
        self.agent.save_model(save_file)

        print("Uploading the final model to wandb...")
        wandb_file = self._upload_model(str(save_file))
        relative_elo = self.compute_tournament_metrics(wandb_file)

        if self.best_model_relative_elo > relative_elo and self.best_model_filename is not None:
            print("Uploading the best model to wandb...")
            self._upload_model(self.best_model_filename, aliases=[f"{self.run.id}-best"])

        wandb.finish()

    def compute_tournament_metrics(self, model_filename: str) -> int:
        _, elo_table, relative_elo, win_perc, absolute_elo, dumb_score = self.metrics.compute(
            self.agent_encoded_name + f",model_filename={model_filename}"
        )

        print(f"Tournament Metrics - Relative elo: {relative_elo}, win percentage: {win_perc}")
        if relative_elo > self.best_model_relative_elo:
            self.best_model_relative_elo = relative_elo
            self.best_model_filename = model_filename
            print("Best Relative Elo so far!")

        wandb_elo_table = wandb.Table(
            columns=["Player", "elo"], data=[[player, elo] for player, elo in elo_table.items()]
        )
        self.run.log(
            {
                "elo": wandb_elo_table,
                "relative_elo": relative_elo,
                "win_perc": win_perc,
                "absolute_elo": absolute_elo,
                "dumb_score": dumb_score,
            },
            step=self.episode_count,
        )

        return relative_elo
