"""Uses Stable-Baselines3 to train agents in the Quoridor environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Taken from Author: Elliot (https://github.com/elliottower)

TODO:
 - Make it work without flattening the observation space
 - Write down necessary hack for this to work at all or find a better solution
 - (future) Implement a Maskable DQN
"""

import glob
import os
import time

import quoridor_env
import torch
from agents.sb3_ppo import SB3ActionMaskWrapper
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.torch_layers import FlattenExtractor

import wandb
from deep_quoridor.src.agents.sb3_ppo import SB3PPOAgent
from wandb.integration.sb3 import WandbCallback


class DictFlattenExtractor(FlattenExtractor):
    """
    This class is necessary because the default FlattenExtractor does not work with dict spaces.
    It just tries to call "flatten" directly which of course doesn't exist on a dict.
    NOTE: It is important to be careful to not flatten the batch dimension (first one).
    """

    def __init__(self, observation_space: spaces.Box):
        super().__init__(observation_space)

    def forward(self, obs: dict) -> torch.Tensor:
        thobs = torch.tensor([], device=list(obs.values())[0].device)
        for v in obs.values():
            thobs = torch.cat((thobs, torch.tensor(v).flatten(start_dim=1)), dim=1)
        return thobs


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn.env(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    # Custom wrapper to convert PettingZoo envs to work with SB3 action masking
    env = SB3ActionMaskWrapper(env)

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    wandb.init(project="deep_quoridor", job_type="train", config=env_kwargs, sync_tensorboard=True)

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    policy_kwargs = {"features_extractor_class": DictFlattenExtractor}
    model = MaskablePPO(MaskableActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs)
    model.set_random_seed(seed)
    model.learn(
        total_timesteps=steps,
        callback=WandbCallback(gradient_save_freq=1000),
    )

    model_id = SB3PPOAgent(**env_kwargs).model_id()
    local_filename = f"{model_id}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(local_filename)
    artifact = wandb.Artifact(f"{model_id}", type="model")
    artifact.add_file(local_path=local_filename)
    artifact.save()
    wandb.finish()

    print(f"Model {model_id} has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn.env(render_mode=render_mode, **env_kwargs)
    env = SB3ActionMaskWrapper(env)

    print(f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[1]}.")

    try:
        model_id = SB3PPOAgent(**env_kwargs).model_id()
        latest_policy = max(glob.glob(f"{model_id}*.zip"), key=os.path.getctime)
    except ValueError:
        print("Policy not found.")
        exit(0)

    model = MaskablePPO.load(latest_policy)

    scores = {agent: 0 for agent in env.possible_agents}
    total_rewards = {agent: 0 for agent in env.possible_agents}
    round_rewards = []

    for i in range(num_games):
        env.reset(seed=i)
        # env.action_space(env.possible_agents[0]).seed(i)

        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()

            # Separate observation and action mask
            action_mask = env.action_mask()

            if termination or truncation:
                # If there is a winner, keep track, otherwise don't change the scores (tie)
                if env.rewards[env.possible_agents[0]] != env.rewards[env.possible_agents[1]]:
                    winner = max(env.rewards, key=env.rewards.get)
                    scores[winner] += env.rewards[winner]  # only tracks the largest reward (winner of game)
                # Also track negative and positive rewards (penalizes illegal moves)
                for a in env.possible_agents:
                    total_rewards[a] += env.rewards[a]
                # List of rewards by round, for reference
                round_rewards.append(env.rewards)
                break
            else:
                if agent == env.possible_agents[0]:
                    space = env.action_space
                    act = space.sample(action_mask)
                else:
                    # Note: PettingZoo expects integer actions # TODO: change chess to cast actions to type int?
                    act = int(model.predict(observation, action_masks=action_mask, deterministic=True)[0])
            env.step(act)
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[1]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


if __name__ == "__main__":
    env_fn = quoridor_env

    env_kwargs = {"board_size": 5, "max_walls": 3}

    # Evaluation/training hyperparameter notes:
    # 10k steps: Winrate:  0.76, loss order of 1e-03
    # 20k steps: Winrate:  0.86, loss order of 1e-04
    # 40k steps: Winrate:  0.86, loss order of 7e-06

    # Train a model against itself (takes ~20 seconds on a laptop CPU)
    train_action_mask(env_fn, steps=20_480, seed=0, **env_kwargs)

    # Evaluate 100 games against a random agent (winrate should be ~80%)
    eval_action_mask(env_fn, num_games=100, render_mode=None, **env_kwargs)

    # Watch two games vs a random agent
    eval_action_mask(env_fn, num_games=2, render_mode="human", **env_kwargs)
