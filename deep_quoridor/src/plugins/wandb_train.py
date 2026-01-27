import datetime
import getpass
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import wandb
import wandb.wandb_run
from agent_evolution_tournament import AgentEvolutionTournament
from agents.core.trainable_agent import TrainableAgent
from arena_utils import ArenaPlugin
from metrics import Metrics
from utils import Timer, resolve_path
from utils.subargs import SubargsBase, override_subargs

# Prevents getting messages in the console every few lines telling you to install weave
os.environ["WANDB_DISABLE_WEAVE"] = "true"


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

    # When to upload models to WandB. "never", "last", or "always"
    #
    # If Set to "last", we actually upload every model but each time we do we delete the previously uploaded
    # model for this run. This means that even if the run is terminated early, we should have to most recent one.
    model_upload_policy: str = "last"

    # How often to log training metrics
    log_every: int = 10

    # Wether workers will also log to wandb (in separate runs)
    log_from_workers: bool = True

    def run_id(self):
        return f"{self.prefix}-{self.suffix}"


class WandbTrainPlugin(ArenaPlugin):
    def __init__(
        self,
        params: WandbParams,
        total_episodes: int,
        agent_encoded_name: str,
        metrics: Metrics,
        agent_evolution_tournament: Optional[AgentEvolutionTournament] = None,
        include_raw_metrics: bool = False,
        wandb_run: wandb.wandb_run.Run | None = None,
    ):
        self.params = params
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.agent = None
        self.agent_encoded_name = agent_encoded_name
        self.best_model_filename = None
        # Notice that the best model won't be uploaded if it's not better than the initialization.
        self.best_model_relative_elo = -800
        self.metrics = metrics
        self.agent_evolution_tournament = agent_evolution_tournament
        self.include_raw_metrics = include_raw_metrics

        if wandb_run is None:
            self.run = wandb.init(
                project=self.params.project,
                job_type="train",
                id=self.params.run_id(),
                group=f"{self.params.run_id()}",
                notes=self.params.notes,
            )
        else:
            self.run = wandb_run
        Timer.set_wandb_run(self.run)

        self._initialized = False
        self._last_uploaded_model = None

    def _initialize(
        self,
        game,
    ):
        if self._initialized:
            return

        assert self.agent is not None
        assert self.run is not None

        config = {
            "board_size": game.board_size,
            "max_walls": game.max_walls,
            "episodes": self.total_episodes,
            "player_args": self.agent.params,
        }
        config.update(self.agent.model_hyperparameters())

        self.run.config.update(config)
        assert self.run.tags is not None
        self.run.tags = self.run.tags + (self.agent.model_id(), f"-{self.params.run_id()}")

        wandb.define_metric("Loss step", hidden=True)
        wandb.define_metric("Epoch", hidden=True)
        wandb.define_metric("Episode", hidden=True)
        wandb.define_metric("Move num", hidden=True)
        wandb.define_metric("game_diversity_*", "Move num")
        wandb.define_metric("loss_*", "Loss step")
        wandb.define_metric("*", "Episode")

        self._initialized = True

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
                {"loss": avg_loss, "reward": avg_reward, "epsilon": self.agent.epsilon, "Episode": self.episode_count}
            )

    def upload_model(self, model_file: str, extra_files: list[str] = []) -> str | None:
        if self.params.model_upload_policy == "never":
            return None

        assert self.agent
        Timer.start("upload_model")
        artifact = wandb.Artifact(f"{self.agent.model_id()}", type="model", metadata=asdict(self.agent.params))
        artifact.add_file(local_path=model_file)
        for file in extra_files:
            artifact.add_file(local_path=file)

        artifact.save()
        logged_artifact = wandb.log_artifact(artifact)
        logged_artifact.wait(300)
        logged_artifact.aliases.extend([f"ep_{self.episode_count}-{self.run.id}"])
        logged_artifact.save()

        if self.params.model_upload_policy == "last" and self._last_uploaded_model is not None:
            # Only keep the most recent model (and associated files)
            self._last_uploaded_model.delete(delete_aliases=True)
        self._last_uploaded_model = logged_artifact

        print(f"Done! Model uploaded with version {logged_artifact.version} and aliases {logged_artifact.aliases}")

        wand_file = resolve_path(self.agent.params.wandb_dir, self.agent.wandb_local_filename(artifact))

        # Now that we know the digest, copy the file to the wandb dir and include the digest, so it takes
        # the expected name and doesn't need to be re-downloaded from wandb.
        # Source and target file are in the same path, just a different file name
        os.makedirs(Path(wand_file).absolute().parents[0], exist_ok=True)
        shutil.copy(model_file, wand_file)
        print(f"Model saved to {wand_file}")
        Timer.finish("upload_model", self.episode_count)

        return str(wand_file)

    def end_arena(self, game, results):
        wandb.finish()

    def _run_benchmark(self, agent_encoded_name: str, prefix: str = ""):
        if prefix != "":
            prefix = prefix + "_"

        Timer.start(f"{prefix}benchmark")
        (
            _,
            _,
            relative_elo,
            win_perc,
            p1_stats,
            p2_stats,
            absolute_elo,
            dumb_score,
        ) = self.metrics.compute(agent_encoded_name)
        Timer.finish(f"{prefix}benchmark", self.episode_count)

        metrics = {
            f"{prefix}relative_elo": relative_elo,
            f"{prefix}win_perc": win_perc,
            f"{prefix}absolute_elo": absolute_elo,
            f"{prefix}dumb_score": dumb_score,
            "Episode": self.episode_count,  # x axis
        }

        metrics.update(self.metrics.metrics_from_stats(prefix, p1_stats, p2_stats))

        return metrics

    def compute_tournament_metrics(self, model_filename: str) -> int:
        agent_name = self.agent_encoded_name.split(":")[0]
        override_args = {"model_filename": model_filename, "nick": f"{agent_name}_{self.episode_count}"}
        agent_encoded_name = override_subargs(self.agent_encoded_name, override_args)

        metrics = self._run_benchmark(agent_encoded_name)
        relative_elo = metrics["relative_elo"]
        win_perc = metrics["win_perc"]
        print(f"Tournament Metrics - Relative elo: {relative_elo}, win percentage: {win_perc}")
        if relative_elo > self.best_model_relative_elo:
            self.best_model_relative_elo = relative_elo
            self.best_model_filename = model_filename
            print("Best Relative Elo so far!")

        if self.include_raw_metrics:
            raw_encoded_name = override_subargs(agent_encoded_name, {"mcts_n": 0})
            raw_metrics = self._run_benchmark(raw_encoded_name, prefix="raw")
            metrics.update(raw_metrics)

        if self.agent_evolution_tournament is not None:
            Timer.start("benchmark-agent-evolution")
            elos = self.agent_evolution_tournament.add_agent_and_compute(agent_encoded_name)
            Timer.finish("benchmark-agent-evolution", self.episode_count)
            elos_by_agent_episode = {}
            for nick, elo in elos.items():
                metrics[f"agent_evolution_{nick}"] = int(elo)
                episode = int(nick.split("_")[-1])
                elos_by_agent_episode[episode] = int(elo)

            sorted_elos = sorted(elos_by_agent_episode.items(), key=lambda x: x[1], reverse=True)
            for i, (ep, _) in enumerate(sorted_elos):
                metrics[f"agent_evolution_place_{i + 1}"] = ep

        self.run.log(metrics)

        return relative_elo
