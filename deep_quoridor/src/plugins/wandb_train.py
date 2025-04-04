import os
from dataclasses import asdict

from agents.core.trainable_agent import AbstractTrainableAgent
from arena_utils import ArenaPlugin

import wandb


class WandbTrainPlugin(ArenaPlugin):
    def __init__(self, update_every: int, total_episodes: int, agent: AbstractTrainableAgent, path: str):
        self.agent = agent
        self.update_every = update_every
        self.total_episodes = total_episodes
        self.episode_count = 0
        self.path = path

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
        save_file = os.path.join(self.path, f"{self.agent.model_id()}.pt")
        self.agent.save_model(save_file)

        artifact = wandb.Artifact(f"{self.agent.model_id()}", type="model", metadata=asdict(self.agent.params))
        artifact.add_file(local_path=save_file)
        artifact.save()

        wand_file = os.path.join(self.path, self.agent.wandb_local_filename(artifact))

        os.rename(save_file, wand_file)
