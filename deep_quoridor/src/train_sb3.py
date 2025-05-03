"""Uses Stable-Baselines3 to train agents in the Quoridor environment using invalid action masking.

For information about invalid action masking in PettingZoo, see https://pettingzoo.farama.org/api/aec/#action-masking
For more information about invalid action masking in SB3, see https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html

Taken from Author: Elliot (https://github.com/elliottower)

TODO:
 - Make it work without flattening the observation space
 - Write down necessary hack for this to work at all or find a better solution
 - (future) Implement a Maskable DQN
"""

import argparse
import datetime
import glob
import os
import time

import quoridor_env
import torch
from agents.sb3_ppo import DictFlattenExtractor, SB3PPOAgent, make_env_fn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from utils import set_deterministic

import wandb
from wandb.integration.sb3 import WandbCallback


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


def train_action_mask(env_fn, steps=10_000, seed=0, upload_to_wandb=True, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    # Initialize wandb with sync_tensorboard to log all SB3 TensorBoard metrics
    wandb.init(project="deep_quoridor", job_type="train", config=env_kwargs, sync_tensorboard=True)

    env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
    # MaskablePPO behaves the same as SB3's PPO unless the env is wrapped
    # with ActionMasker. If the wrapper is detected, the masks are automatically
    # retrieved and used when learning. Note that MaskablePPO does not accept
    # a new action_mask_fn kwarg, as it did in an earlier draft.
    # Configure MLP policy network architecture for a 5x5 board with 3 max walls
    # Total flattened input features: 84(25 for my_board, 25 for opponent_board, 32 for walls, 1+1 for wall counts)
    policy_kwargs = {
        "features_extractor_class": DictFlattenExtractor,
        # NOTE(adamantivm) These params haven't proven to be any particular good so far
        "net_arch": {
            "pi": [256, 256, 256],
            "vf": [256, 256, 256],
        },
        "activation_fn": torch.nn.ReLU,
    }

    # Configure model with tensorboard logging to ensure metrics are captured
    tensorboard_log = "runs/sb3_tensorboard"
    model = MaskablePPO(
        MaskableActorCriticPolicy, env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_log
    )
    model.set_random_seed(seed)

    # Simplified WandbCallback with verbosity set to capture loss metrics
    model.learn(
        total_timesteps=steps,
        callback=WandbCallback(
            gradient_save_freq=1000,
            verbose=2,  # More verbose logging to capture loss metrics
        ),
    )

    model_id = SB3PPOAgent(**env_kwargs).model_id()
    local_filename = f"{model_id}_{time.strftime('%Y%m%d-%H%M%S')}.zip"
    model.save(local_filename)

    if upload_to_wandb:
        artifact = wandb.Artifact(f"{model_id}", type="model")
        artifact.add_file(local_path=local_filename)
        artifact.save()
        wandb.finish()

    print(f"Model {model_id} has been saved.")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render_mode=None, player=0, **env_kwargs):
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[player]}.")

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
                if agent == env.possible_agents[1 - player]:
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
        winrate = scores[env.possible_agents[player]] / sum(scores.values())
    print("Rewards by round: ", round_rewards)
    print("Total rewards (incl. negative rewards): ", total_rewards)
    print("Winrate: ", winrate)
    print("Final scores: ", scores)
    return round_rewards, total_rewards, winrate, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate SB3 agents in Quoridor using invalid action masking"
    )
    parser.add_argument("-N", "--board_size", type=int, default=3, help="Board Size")
    parser.add_argument("-W", "--max_walls", type=int, default=0, help="Max walls per player")
    parser.add_argument("-e", "--steps", type=int, default=20_480, help="Number of steps to train for")
    parser.add_argument("-g", "--num_games", type=int, default=100, help="Number of games for evaluation")
    parser.add_argument("-i", "--seed", type=int, default=0, help="Random seed for training and evaluation")
    parser.add_argument("--no-train", action="store_true", default=False, help="Skip training and only run evaluation")
    parser.add_argument("--no-upload", action="store_true", default=False, help="Skip uploading artifacts to wandb")
    parser.add_argument("--no-eval", action="store_true", default=False, help="Skip evaluation and only run training")
    parser.add_argument(
        "-rp",
        "--run_prefix",
        type=str,
        default="sb3-ppo",
        help="Run prefix to use for this run. This will be used for naming, and tagging artifacts",
    )
    parser.add_argument(
        "-rs",
        "--run_suffix",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Run suffix. Default is current date and time. This will be used for naming, and tagging artifacts",
    )

    args = parser.parse_args()

    env_fn = make_env_fn(quoridor_env.env)
    env_kwargs = {"board_size": args.board_size, "max_walls": args.max_walls}

    # Set random seed for reproducibility
    set_deterministic(args.seed)

    print("\nRunning SB3 training/evaluation with:")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Run ID: {args.run_prefix}-{args.run_suffix}")

    # Train a model against itself
    if not args.no_train:
        print(f"\nTraining for {args.steps} steps with seed {args.seed}...")
        train_action_mask(env_fn, steps=args.steps, seed=args.seed, upload_to_wandb=not args.no_upload, **env_kwargs)

    # Evaluate games against a random agent
    if not args.no_eval:
        print(f"\nEvaluating {args.num_games} games against a random agent...")
        eval_action_mask(env_fn, num_games=args.num_games // 2, render_mode=None, player=0, **env_kwargs)
        eval_action_mask(env_fn, num_games=args.num_games // 2, render_mode=None, player=1, **env_kwargs)
