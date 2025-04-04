import os
from dataclasses import asdict

from agents.core.trainable_agent import AbstractTrainableAgent
from arena_utils import ArenaPlugin
from utils.misc import resolve_path

import wandb


class WandbTrainPlugin(ArenaPlugin):
    def __init__(self, update_every: int, total_episodes: int, agent: AbstractTrainableAgent):
        self.agent = agent
        self.update_every = update_every
        self.total_episodes = total_episodes
        self.episode_count = 0

    def start_arena(self, game, total_games: int):
        config = {
            "board_size": game.board_size,
            "max_walls": game.max_walls,
            "episodes": self.total_episodes,
            "player_args": self.agent.params,
        }
        config.update(self.agent.model_hyperparameters())

        self.run = wandb.init(
            project="deep_quoridor",
            job_type="train",
            config=config,
            tags=[self.agent.model_id()],
        )

    def end_game(self, game, result):
        if self.episode_count % self.update_every == 0 or self.episode_count == (self.total_episodes - 1):
            avg_reward, avg_loss = self.agent.compute_loss_and_reward(self.update_every)

            self.run.log(
                {"loss": avg_loss, "reward": avg_reward, "epsilon": self.agent.epsilon},
                step=self.episode_count,
            )
        self.episode_count += 1

    def end_arena(self, game, results):
        # Save the model in the wandb directory with the suffix "temp".  The file will be renamed
        # once we upload it to wandb and have the digest.
        save_file = resolve_path(self.agent.params.wandb_dir, self.agent.resolve_filename("temp"))
        self.agent.save_model(save_file)

        artifact = wandb.Artifact(f"{self.agent.model_id()}", type="model", metadata=asdict(self.agent.params))
        artifact.add_file(local_path=str(save_file))
        artifact.save()

        wand_file = resolve_path(self.agent.params.wandb_dir, self.agent.wandb_local_filename(artifact))

        # Now that we know the digest, rename the file to include it.  Source and target file are in
        # the same path, just a different file name
        os.rename(save_file, wand_file)
        print(f"Model saved to {wand_file}")

        wandb.finish()
