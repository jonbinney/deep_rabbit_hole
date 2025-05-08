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
from typing import override

import quoridor_env
import torch
from agents import AgentRegistry
from agents.sb3_ppo import DictFlattenExtractor, SB3PPOAgent, make_env_fn
from arena import Arena
from renderers import ArenaResultsRenderer
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from utils import resolve_path, set_deterministic

import wandb
from wandb.integration.sb3 import WandbCallback


def mask_fn(env):
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.action_mask()


class SwapPlayerCallback(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.current_player = "player_0"

    @override
    def _on_rollout_start(self) -> None:
        print(f"Playing as {self.current_player}")

    @override
    def _on_rollout_end(self):
        if self.current_player == "player_0":
            self.current_player = "player_1"
        else:
            self.current_player = "player_0"
        self.env.set_player(self.current_player)

    @override
    def _on_step(self):
        return True


def train_action_mask(env_fn, steps=10_000, seed=0, upload_to_wandb=False, train_kwargs={}, **env_kwargs):
    """Train a single model to play as each agent in a zero-sum game environment using invalid action masking."""
    env = env_fn(**env_kwargs)

    print(f"Starting training on {str(env.metadata['name'])}.")

    env.reset(seed=seed)  # Must call reset() in order to re-define the spaces

    # Initialize wandb with sync_tensorboard to log all SB3 TensorBoard metrics
    wandb.init(
        project="deep_quoridor",
        job_type="train",
        config={**env_kwargs, **train_kwargs},
        sync_tensorboard=True,
    )

    masked_env = ActionMasker(env, mask_fn)  # Wrap to enable masking (SB3 function)
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
        MaskableActorCriticPolicy,
        masked_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        # n_steps=10000,
        # learning_rate=3e-3,
    )
    model.set_random_seed(seed)

    # Simplified WandbCallback with verbosity set to capture loss metrics
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        verbose=2,
    )

    swap_player_callback = SwapPlayerCallback(env)

    # Play as player 1
    model.learn(
        total_timesteps=steps,
        callback=CallbackList([wandb_callback, swap_player_callback]),
    )

    # Create an SB3PPO agent to handle model file naming
    sb3_agent = SB3PPOAgent(**env_kwargs)
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Use the utils.resolve_path to get the standard models directory path
    # and make sure it exists
    models_dir = resolve_path(sb3_agent.params.model_dir)
    os.makedirs(models_dir, exist_ok=True)

    # Use the agent's resolve_filename method to generate standard filenames
    # Save both as timestamped version and as 'final' version
    timestamped_filename = sb3_agent.resolve_filename(timestamp)
    final_filename = sb3_agent.resolve_filename("final")

    # Construct full paths using resolve_path
    model_path_timestamped = resolve_path(sb3_agent.params.model_dir, timestamped_filename)
    model_path_final = resolve_path(sb3_agent.params.model_dir, final_filename)

    # Save the model files
    model.save(str(model_path_timestamped))
    model.save(str(model_path_final))

    print(f"Model saved to:\n- {model_path_timestamped} (timestamped version)\n- {model_path_final} (final version)")

    if upload_to_wandb:
        model_id = sb3_agent.model_id()
        artifact = wandb.Artifact(f"{model_id}", type="model")
        artifact.add_file(local_path=str(model_path_timestamped))
        artifact.save()
        wandb.finish()
        print(f"Model uploaded to wandb as artifact: {model_id}")

    print(f"Finished training on {str(env.unwrapped.metadata['name'])}.\n")

    env.close()


def eval_action_mask(env_fn, num_games=100, render=False, player=0, **env_kwargs):
    # Set render_mode to "human" if render=True, otherwise None
    render_mode = "human" if render else None
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(f"Starting evaluation vs a random agent. Trained agent will play as {env.possible_agents[player]}.")

    # Create agent to help with model handling
    sb3_agent = SB3PPOAgent(**env_kwargs)

    try:
        # First try to use the _final model we just saved
        final_path = resolve_path(sb3_agent.params.model_dir, sb3_agent.resolve_filename("final"))
        if os.path.exists(final_path):
            model = MaskablePPO.load(str(final_path))
            print(f"Using final model: {final_path}")
        else:
            # Fallback to any timestamped model in the models directory
            models_dir = resolve_path(sb3_agent.params.model_dir)
            pattern = os.path.join(str(models_dir), f"{sb3_agent.model_id()}*.zip")
            latest_policy = max(glob.glob(pattern), key=os.path.getctime)
            model = MaskablePPO.load(latest_policy)
            print(f"Using latest model: {latest_policy}")
    except (ValueError, FileNotFoundError):
        print("Policy not found in the models directory.")
        exit(0)

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

            if render_mode == "human":
                print(env.render())
    env.close()

    # Avoid dividing by zero
    if sum(scores.values()) == 0:
        winrate = 0
    else:
        winrate = scores[env.possible_agents[player]] / sum(scores.values())
    # print("Rewards by round: ", round_rewards)
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
    parser.add_argument("-m", "--rewards_multiplier", type=int, default=1, help="Multiplier for rewards")
    parser.add_argument("-i", "--seed", type=int, default=0, help="Random seed for training and evaluation")
    parser.add_argument("--no-train", action="store_true", default=False, help="Skip training and only run evaluation")
    parser.add_argument("--upload", action="store_true", default=False, help="Upload artifacts to wandb")
    parser.add_argument("--no-eval", action="store_true", default=False, help="Skip evaluation and only run training")
    parser.add_argument("--render", action="store_true", default=False, help="Render environment during evaluation")
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
    parser.add_argument(
        "-o",
        "--opponent",
        type=str,
        default=None,
        # choices=AgentRegistry.names() + [None],
        help=f"Opponent agent type. Available options: {AgentRegistry.names()} or None for self-play",
    )

    args = parser.parse_args()

    temp_env = quoridor_env.env(board_size=args.board_size, max_walls=args.max_walls)
    env_kwargs = {
        "board_size": args.board_size,
        "max_walls": args.max_walls,
        "action_space": temp_env.action_space(None),
        "observation_space": temp_env.observation_space(None),
    }
    train_kwargs = {
        "steps": args.steps,
        "seed": args.seed,
        "opponent": args.opponent,
        "rewards_multiplier": args.rewards_multiplier,
    }

    opponent = None
    if args.opponent is not None:
        opponent = AgentRegistry.create_from_encoded_name(args.opponent, **env_kwargs)

    env_fn = make_env_fn(quoridor_env.env, opponent=opponent, rewards_multiplier=args.rewards_multiplier)

    # Set random seed for reproducibility
    set_deterministic(args.seed)

    print("\nRunning SB3 training/evaluation with:")
    print(f"Board size: {args.board_size}x{args.board_size}")
    print(f"Max walls: {args.max_walls}")
    print(f"Run ID: {args.run_prefix}-{args.run_suffix}")

    # Train a model against opponent or itself
    if not args.no_train:
        print(f"\nTraining for {args.steps} steps with seed {args.seed}...")
        print(f"Opponent: {args.opponent if args.opponent else 'Self-play'}")
        train_action_mask(
            env_fn,
            steps=args.steps,
            seed=args.seed,
            upload_to_wandb=args.upload,
            train_kwargs=train_kwargs,
            **env_kwargs,
        )

    # Evaluate games against the opponent
    if not args.no_eval:
        # Create a new env fn without the opponent, for evaluation
        env_fn = make_env_fn(quoridor_env.env)
        print(f"\nEvaluating {args.num_games} games against a random agent...")
        eval_action_mask(env_fn, num_games=args.num_games // 2, render=args.render, player=0, **env_kwargs)
        eval_action_mask(env_fn, num_games=args.num_games // 2, render=args.render, player=1, **env_kwargs)

        arena = Arena(
            board_size=args.board_size,
            max_walls=args.max_walls,
            renderers=[ArenaResultsRenderer()],
        )

        arena.play_games(players=["random", "sb3ppo"], times=args.num_games)
