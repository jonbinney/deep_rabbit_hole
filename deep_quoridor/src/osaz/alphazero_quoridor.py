#!/usr/bin/env python3
"""
A script to train an AlphaZero agent to play Quoridor using OpenSpiel.
"""

import collections
import datetime
import getpass
import json
import multiprocessing
import os

# Set TF environment variables for GPU memory management and device visibility.
# This must be done BEFORE TensorFlow is imported.
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Set the start method to 'spawn' for CUDA compatibility and monkey-patch
# to prevent OpenSpiel from overriding it.
try:
    multiprocessing.set_start_method("spawn", force=True)

    # Monkey-patch to prevent OpenSpiel from changing the start method.
    def _do_nothing_set_start_method(method, force=False):
        pass

    multiprocessing.set_start_method = _do_nothing_set_start_method
except RuntimeError:
    # This may be raised in child processes where the context is already set.
    pass

import numpy as np
import pyspiel
from absl import app, flags
from agents.alphazero_os import AlphaZeroOSAgent, AlphaZeroOSParams
from open_spiel.python.algorithms import mcts
from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator
from open_spiel.python.algorithms.alpha_zero import model as az_model
from open_spiel.python.bots import uniform_random
from open_spiel.python.utils import spawn
from osaz.wandb_logger import AlphaZeroWandbLogger

import wandb

# Import wandb for logging and visualization is at the top


# Add a custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


# Monkey patch the json module to use our encoder by default
json._default_encoder = NumpyEncoder()

FLAGS = flags.FLAGS

# Game parameters
flags.DEFINE_integer("board_size", 5, "Board size for Quoridor (e.g., 5 for 5x5).")
flags.DEFINE_integer("num_players", 2, "Number of players in the game.")
flags.DEFINE_integer("wall_count", 3, "Number of walls per player.")

# Training parameters
flags.DEFINE_integer("num_iterations", 10, "Number of training iterations.")
flags.DEFINE_integer("num_train_episodes", 100, "Number of self-play episodes per iteration.")
flags.DEFINE_integer("save_checkpoint_iter", 5, "Save model every N iterations.")
flags.DEFINE_string("model_path", "models/osaz", "Path to save/load the model.")
flags.DEFINE_bool("load_model", False, "Whether to load an existing model.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_integer("train_batch_size", 128, "Batch size for training.")
flags.DEFINE_integer("replay_buffer_size", 10000, "Size of the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 3, "How many times to reuse each sample.")
flags.DEFINE_float("learning_rate_decay", 0.1, "Learning rate decay rate.")
flags.DEFINE_integer("evaluation_window", 100, "Window size for win rate evaluation.")
flags.DEFINE_float("uct_c", 2.0, "UCT exploration constant.")
flags.DEFINE_integer("max_simulations", 100, "Maximum number of MCTS simulations per move.")
flags.DEFINE_float("policy_alpha", 1.0, "Dirichlet noise alpha parameter.")
flags.DEFINE_float("policy_epsilon", 0.25, "Dirichlet noise epsilon parameter.")
flags.DEFINE_float("temperature", 1.0, "Temperature for move selection.")
flags.DEFINE_integer("temperature_drop", 15, "Drop temperature to 0 after this many moves.")
flags.DEFINE_string("nn_model", "mlp", "Neural network model architecture.")
flags.DEFINE_integer("nn_width", 1024, "Width of the neural network.")
flags.DEFINE_integer("nn_depth", 4, "Depth of the neural network.")
flags.DEFINE_integer("eval_levels", 7, "Number of evaluation levels.")
flags.DEFINE_integer("actors", 2, "Number of actor processes to run in parallel.")
flags.DEFINE_integer("evaluators", 2, "Number of evaluation processes to run in parallel.")

# Evaluation parameters
flags.DEFINE_integer("eval_games", 10, "Number of games for evaluation.")
flags.DEFINE_boolean("eval_only", False, "Skip training and only evaluate an existing model.")
flags.DEFINE_bool("verbose", False, "Be verbose.")

# Weights & Biases parameters
flags.DEFINE_string("wandb_project", "deep_quoridor", "Name of the wandb project.")
flags.DEFINE_string("wandb_run_prefix", getpass.getuser(), "User-defined prefix for the wandb run name.")
flags.DEFINE_string(
    "wandb_run_suffix", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), "Suffix for the wandb run name."
)
flags.DEFINE_string("wandb_notes", "", "Optional notes for the wandb run.")
flags.DEFINE_bool("use_wandb", True, "Whether to use wandb for logging.")
flags.DEFINE_bool("upload_model", True, "Whether to upload the model to wandb.")
flags.DEFINE_integer("log_interval", 1, "How often to log to wandb during training (every N iterations).")


def main(_):
    """Trains an AlphaZero agent to play Quoridor and evaluates against a random agent."""
    # Initialize wandb if enabled
    wandb_run = None
    if FLAGS.use_wandb:
        run_id = f"{FLAGS.wandb_run_prefix}-{FLAGS.wandb_run_suffix}"
        # Configure wandb and initialize run
        wandb_run = wandb.init(
            project=FLAGS.wandb_project,
            name=run_id,
            id=run_id,
            config={
                # Game parameters
                "board_size": FLAGS.board_size,
                "num_players": FLAGS.num_players,
                "wall_count": FLAGS.wall_count,
                # Training parameters
                "num_iterations": FLAGS.num_iterations,
                "num_train_episodes": FLAGS.num_train_episodes,
                "learning_rate": FLAGS.learning_rate,
                "weight_decay": FLAGS.weight_decay,
                "train_batch_size": FLAGS.train_batch_size,
                "replay_buffer_size": FLAGS.replay_buffer_size,
                "replay_buffer_reuse": FLAGS.replay_buffer_reuse,
                "learning_rate_decay": FLAGS.learning_rate_decay,
                "uct_c": FLAGS.uct_c,
                "max_simulations": FLAGS.max_simulations,
                "policy_alpha": FLAGS.policy_alpha,
                "policy_epsilon": FLAGS.policy_epsilon,
                "temperature": FLAGS.temperature,
                "nn_model": FLAGS.nn_model,
                "nn_width": FLAGS.nn_width,
                "nn_depth": FLAGS.nn_depth,
            },
            notes=FLAGS.wandb_notes,
            tags=["alphazero", "quoridor", "openspiel"],
        )

    # Format the game string with parameters to match OpenSpiel's requirements
    game_string = f"quoridor(board_size={FLAGS.board_size},players={FLAGS.num_players},wall_count={FLAGS.wall_count})"

    # Create the actual game instance for evaluation purposes
    game = pyspiel.load_game(
        "quoridor", {"board_size": FLAGS.board_size, "players": FLAGS.num_players, "wall_count": FLAGS.wall_count}
    )

    # Based on our observation_test.py, Quoridor uses a flat tensor representation
    # For a 5x5 board with 2 players, the observation tensor shape is (405,)
    # For AlphaZero with MLP, we need to set observation_shape to None to let it infer the shape
    observation_shape = None

    # Estimate output size based on action space
    # For quoridor, actions include:
    # - pawn moves (up to 4 directions from any position) = board_size^2 * 4
    # - wall placements (horizontal or vertical at each intersection) = (board_size-1)^2 * 2
    output_size = (FLAGS.board_size**2) * 4 + ((FLAGS.board_size - 1) ** 2) * 2

    # Create the AlphaZero config
    config = alpha_zero.Config(
        game=game_string,  # Use the game string, not the game object
        path=FLAGS.model_path,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        train_batch_size=FLAGS.train_batch_size,
        replay_buffer_size=FLAGS.replay_buffer_size,
        replay_buffer_reuse=FLAGS.replay_buffer_reuse,
        max_steps=FLAGS.num_iterations * FLAGS.num_train_episodes,
        checkpoint_freq=FLAGS.save_checkpoint_iter * FLAGS.num_train_episodes,
        actors=FLAGS.actors,
        evaluators=FLAGS.evaluators,
        evaluation_window=FLAGS.evaluation_window,
        eval_levels=FLAGS.eval_levels,
        uct_c=FLAGS.uct_c,
        max_simulations=FLAGS.max_simulations,
        policy_alpha=FLAGS.policy_alpha,
        policy_epsilon=FLAGS.policy_epsilon,
        temperature=FLAGS.temperature,
        temperature_drop=FLAGS.temperature_drop,
        nn_model=FLAGS.nn_model,
        nn_width=FLAGS.nn_width,
        nn_depth=FLAGS.nn_depth,
        observation_shape=observation_shape,  # Explicitly set the observation shape
        output_size=output_size,
        quiet=not FLAGS.verbose,
    )

    checkpoint_dir = config.path
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Train the model if not in eval_only mode
    if not FLAGS.eval_only:
        # Initialize the wandb logger thread if wandb is enabled
        wandb_logger = None
        if FLAGS.use_wandb and wandb_run is not None:
            # Create and start the logger thread
            wandb_logger = AlphaZeroWandbLogger(
                experiment_dir=checkpoint_dir,
                wandb_run=wandb_run,
                watch_interval_seconds=30,
            )
            wandb_logger.start()
            print(f"Started wandb metrics logger thread for {checkpoint_dir}/learner.jsonl")
        # Run Alpha Zero with the given configuration
        with spawn.main_handler():
            alpha_zero.alpha_zero(config)

        # Stop the logger thread if it was started
        if wandb_logger is not None:
            wandb_logger.stop()
        final_checkpoint = f"{checkpoint_dir}/checkpoint--1"
        # Log model as artifact if wandb is enabled
        if FLAGS.use_wandb and FLAGS.upload_model and not FLAGS.eval_only:
            # Create a temporary instance of AlphaZeroOSAgent to generate consistent artifact name
            temp_agent = AlphaZeroOSAgent(
                params=AlphaZeroOSParams(),
                board_size=FLAGS.board_size,
                max_walls=FLAGS.wall_count,
                action_space=None,  # Not needed for model_id
            )
            artifact_name = temp_agent.model_id()
            print(f"Uploading model from {final_checkpoint} to wandb as {artifact_name}")
            model_artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"AlphaZero model trained on Quoridor ({FLAGS.board_size}x{FLAGS.board_size}, {FLAGS.wall_count} walls)",
            )
            model_artifact.add_dir(checkpoint_dir)
            wandb_run.log_artifact(model_artifact)
    else:
        # Just use the existing checkpoint for evaluation
        final_checkpoint = f"{checkpoint_dir}/checkpoint--1"
        print(f"Skipping training, evaluating existing model at {final_checkpoint}")

    # Evaluate the model
    print(f"Evaluating model at {final_checkpoint}")
    results = evaluate_bot(game, final_checkpoint, FLAGS.eval_games)
    print("Evaluation results:")
    print(results)
    # Log evaluation results to wandb
    if FLAGS.use_wandb:
        # Log evaluation metrics
        wandb_run.log(
            {
                "eval/trained_first_win_rate": results["trained_first"].get(0, 0),
                "eval/trained_first_loss_rate": results["trained_first"].get(1, 0),
                "eval/trained_first_draw_rate": results["trained_first"].get(2, 0),
                "eval/trained_second_win_rate": results["trained_second"].get(1, 0),
                "eval/trained_second_loss_rate": results["trained_second"].get(0, 0),
                "eval/trained_second_draw_rate": results["trained_second"].get(2, 0),
            }
        )
        # Create and log a wandb Table with evaluation results
        columns = ["Position", "Win Rate", "Loss Rate", "Draw Rate"]
        data = [
            [
                "First",
                results["trained_first"].get(0, 0),
                results["trained_first"].get(1, 0),
                results["trained_first"].get(2, 0),
            ],
            [
                "Second",
                results["trained_second"].get(1, 0),
                results["trained_second"].get(0, 0),
                results["trained_second"].get(2, 0),
            ],
        ]
        results_table = wandb.Table(columns=columns, data=data)
        wandb_run.log({"eval_results_table": results_table})
        # Finish the wandb run
        wandb_run.finish()

    return 0


def evaluate_bot(game, checkpoint_path, num_games=10):
    """Evaluate a trained bot against a random bot."""
    # Create the trained bot from checkpoint
    trained_bot = create_bot_from_checkpoint(checkpoint_path, game)

    results = {}

    # Trained Agent vs Random (trained agent goes first)
    trained_first = play_games(
        game, trained_bot, uniform_random.UniformRandomBot(1, np.random.RandomState(42)), num_games
    )
    print(f"Trained Bot (P0) vs Random (P1): {trained_first}")
    results["trained_first"] = trained_first

    # Random vs Trained Agent (trained agent goes second)
    trained_second = play_games(
        game, uniform_random.UniformRandomBot(0, np.random.RandomState(42)), trained_bot, num_games
    )
    print(f"Random (P0) vs Trained Bot (P1): {trained_second}")
    results["trained_second"] = trained_second

    return results


def create_bot_from_checkpoint(checkpoint_path, game):
    """Creates an AlphaZero MCTS bot using a checkpoint."""
    # Load the model from checkpoint
    model = az_model.Model.from_checkpoint(checkpoint_path)

    # Create the AlphaZero evaluator with the loaded model
    evaluator = az_evaluator.AlphaZeroEvaluator(game, model)

    # Create the MCTS bot with the AlphaZero evaluator
    bot = mcts.MCTSBot(
        game,
        FLAGS.uct_c,
        FLAGS.max_simulations,
        evaluator,
        random_state=np.random.RandomState(42),
        solve=False,
        verbose=FLAGS.verbose,
    )
    return bot


def evaluate_against_random(game, trained_bot, num_games):
    """Evaluate a trained bot against a random bot."""
    results = {}

    # Trained Agent vs Random (trained agent goes first)
    trained_first = play_games(
        game, trained_bot, uniform_random.UniformRandomBot(1, np.random.RandomState(42)), num_games
    )
    print(f"Trained Bot (P0) vs Random (P1): {trained_first}")
    results["trained_first"] = trained_first

    # Random vs Trained Agent (trained agent goes second)
    trained_second = play_games(
        game, uniform_random.UniformRandomBot(0, np.random.RandomState(42)), trained_bot, num_games
    )
    print(f"Random (P0) vs Trained Bot (P1): {trained_second}")
    results["trained_second"] = trained_second

    return results


def play_games(game, bot_1, bot_2, num_games):
    """Plays a series of games between two bots and returns win rates.

    Includes negative rewards when the agent loses, as per previous learning.
    """
    # If we're not using a perfect information game, use an observation bot.
    # We want the policy to maximize the win rate for player 0, which is the first player.
    results = collections.defaultdict(int)
    for game_num in range(num_games):
        players = [bot_1, bot_2]
        state = game.new_initial_state()
        while not state.is_terminal():
            player = state.current_player()
            action = players[player].step(state)
            state.apply_action(action)

        # Record results
        returns = state.returns()
        if returns[0] > returns[1]:
            results[0] += 1  # Bot 1 wins
        elif returns[1] > returns[0]:
            results[1] += 1  # Bot 2 wins
        else:
            results[2] += 1  # Draw

    return {win_flag: count / num_games for win_flag, count in results.items()}


if __name__ == "__main__":
    app.run(main)
