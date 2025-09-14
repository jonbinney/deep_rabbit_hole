import os
import pickle
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
from quoridor import ActionEncoder, MoveAction, construct_game_from_observation
from utils import get_initial_random_seed, my_device
from utils.subargs import SubargsBase

from agents.alphazero.mcts import MCTS, QuoridorKey
from agents.alphazero.nn_evaluator import NNEvaluator
from agents.core import TrainableAgent


@dataclass
class AlphaZeroParams(SubargsBase):
    training_mode: bool = False

    # After how many self play games we train the network If set to None, agent will not run the
    # NN training itself even if traning_mode == True. This is useful if we want the agent to
    # use params as if it is training, but have the actual NN training run by an higher level
    # function.
    train_every: Optional[int] = 10

    # Learning rate to use for the optimizer
    learning_rate: float = 0.001

    # L2 regularization weight decay coefficient. Weight decay and L2 regularization are the
    # same thing for simple NN optimizers like SGD. For Adam, they are subtly different
    # because of the adaptive learning rates. It seems that the common (best?) practice is to
    # use weight decay with "decoupled weights" to achieve the same effect as L2 regularization.
    # We use the AdamW otimizer which does this by default. 1e-4 here is the same value uesd in
    # the AlphaZero paper.
    #
    # For more explanation weight decay vs L2 regularization, see:
    #
    #     https://www.johntrimble.com/posts/weight-decay-is-not-l2-regularization/
    #
    weight_decay: float = 0.0001

    # Exploration vs exploitation.  0 is pure exploitation, infinite is random exploration.
    temperature: float = 1.0

    # How many moves to remember. The training batches are sampled from this replay buffer.
    # If set to none, the buffer grows without bound.
    replay_buffer_size: Optional[int] = 1000

    # Batch size for training
    batch_size: int = 100

    # How many iterations of optimizer each time we update the model
    optimizer_iterations: int = 100

    # Number of MCTS selections
    mcts_n: int = 100

    # If set, the number of MCTS selections is going to be mcts_k * n_actions, where n_actions
    # is the number of actions available.
    mcts_k: Optional[int] = None

    # A higher number favors exploration over exploitation
    mcts_ucb_c: float = 1.4

    # Parameters for adding Dirichlet noise to the policy distribution at the root MCTS node during
    # training. Adding this noise helps keep MCTS from only focusing on a few moves that happen to end
    # in good results early on.
    #
    # According to the Alphago Zero paper, the action priors for the root node are updated according to:
    #
    #     P(s, a) = (1 - epsilon) * p_a + epsilon * eta_alpha
    #
    # Where eta_alpha is drawn from a Dirichlet distribution Dir(alpha). In other words, the
    # policy distribution that MCTS uses is a weighted average of the one created by the evaluator
    # and a completely random distribution. The default value of 0.25 for epsilon means that the
    # evaluator's policy gets a weight of 75% and the completely random Dirichlet distribution gets
    # a weighting of 25%. The value of alpha changes the shape of the completely Dirichlet
    # distribution; 0.03 is the value used in the Alphago Zero paper.
    #
    # In the Alphazero paper, the value for alpha was set differently for different games:
    #
    #    "Dirichlet noise Dir(α) was added to the prior probabilities in the root node; this was scaled
    #     in inverse proportion to the approximate number of legal moves in a typical position, to a
    #     value of α = {0.3, 0.15, 0.03} for chess, shogi and Go respectively."
    #
    # Those values for alpha work out to approximately (10 / <typical number of valid moves>)
    # For Quoridor, the number of possible moves varies _a lot_ depending on whether a player has walls
    # remaining. For example on a 5x5 board in the start position, there are 3 valid movements and 32
    # valid wall placements, for a total of 35 valid moves. Once a player is out of walls, there are a maximum
    # of 5 valid moves (in a case where diagonal jumps are possible.) More often there are 1 to 4 valid moves.
    #
    # mcts_noise_epsilon - may be between 0.0 and 1.0. If it is 0, then no noise is added. If it is 1.0,
    # then the policy from the evaluator is ignored and pure noise is used. In between, you get a weighted
    # average of the evaluator's policy and Dirichlet noise.
    #
    # mcts_noise_alpha  - can be either None (default) or a positive real value. If it is None, then we use
    # an (10 / typical_number_of_valid_moves), where the typical number of valid moves is esimated based
    # on the board size and whether max_walls is 0 or not.
    #
    # NOTE: these parameters are only used in training_mode. Otherwise no Dirichlet noise is added.
    #
    mcts_noise_epsilon: float = 0.25
    mcts_noise_alpha: Optional[float] = None

    # Number of nodes to pre-evaluate in MCTS. This is to take advantage of the GPU
    # being efficient in batching.
    mcts_pre_evaluate_nodes_total: int = 64

    # If wandb_alias is provided, the model will be fetched from wandb using the model_id and the alias
    wandb_alias: Optional[str] = None

    # When loading from wandb, the project name to be used
    wandb_project: str = "deep_quoridor"

    # If a filename is provided, the model will be loaded from disc
    model_filename: Optional[str] = None

    # Directory where wandb models are stored
    wandb_dir: str = "wandbmodels"

    # Directory where local models are stored
    model_dir = "models"

    # If True, the agent will penalize visited states in MCTS to avoid cycling
    penalized_visited_states: bool = False

    # Whether to save the replay buffer to a file.  This can be used to fast-forward the first batch of games
    # (in which case you can only use the first replay buffer) or to use a tool to visualize any of the replay buffers.
    # The options are: "never" | "first" | "always"
    save_replay_buffer: str = "never"

    # What fraction of the total moves will be used for validation during training, or 0.0 to not use a validation set
    validation_ratio: float = 0.0

    @classmethod
    def training_only_params(cls) -> set[str]:
        """
        Returns a set of parameters that are used only during training.
        These parameters should not be used during playing.
        """
        return {
            "training_mode",
            "train_every",
            "learning_rate",
            "weight_decay",
            "optimizer_iterations",
            "batch_size",
            "replay_buffer_size",
            "save_replay_buffer",
            "validation_ratio",
        }


class AlphaZeroAgent(TrainableAgent):
    def __init__(
        self,
        board_size: int,
        max_walls: int,
        max_steps: int = -1,  # -1 means games are never truncated.
        observation_space=None,
        action_space=None,
        evaluator=None,
        params=AlphaZeroParams(),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.params = params
        self.board_size = board_size
        self.max_walls = max_walls
        self.max_steps = max_steps
        self.device = my_device()
        self.visited_states = set()

        self.action_encoder = ActionEncoder(board_size)
        self.evaluator = NNEvaluator(self.action_encoder, self.device)
        if evaluator is None:
            self.evaluator = NNEvaluator(self.action_encoder, self.device)
        else:
            self.evaluator = evaluator

        # Disable dirichlet noise unless we are in training mode.
        mcts_noise_epsilon = params.mcts_noise_epsilon if params.training_mode else 0.0
        # Estimate the typical number of valid moves and use that to choose the alpha parameter of dirichlet noise.
        # (see parameters class for more explanation of how to choose alpha)
        if params.mcts_noise_alpha is None:
            max_valid_wall_actions = 0 if max_walls == 0 else self.action_encoder.num_actions - board_size**2
            typical_num_valid_actions = float(np.mean([3, max_valid_wall_actions]))
            mcts_noise_alpha = 10.0 / typical_num_valid_actions
        else:
            mcts_noise_alpha: float = params.mcts_noise_alpha

        self.mcts = MCTS(
            params.mcts_n,
            params.mcts_k,
            params.mcts_ucb_c,
            mcts_noise_epsilon,
            mcts_noise_alpha,
            self.max_steps,
            self.evaluator,
            self.visited_states,
            params.mcts_pre_evaluate_nodes_total,
        )

        if params.training_mode and params.train_every is not None:
            self.evaluator.train_prepare(
                params.learning_rate, params.batch_size, params.optimizer_iterations, params.weight_decay
            )

        self.episode_count = 0

        # When playing use 0.0 for temperature so we always chose the best available action.
        self.temperature = params.temperature if params.training_mode else 0.0

        self.replay_buffer = deque(maxlen=params.replay_buffer_size)

        # Metrics tracking
        self.recent_losses = []

        # Just to make the training status plugin happy
        self.epsilon = 0.0

        # Avoid circular imports
        from metrics import Metrics

        self.metrics = Metrics(board_size, max_walls)

        self.first_replay_buffer_saved = False
        self.first_replay_buffer_loaded = False

        if self.load_replay_buffer_from_file():
            # Run initial training if we have the saved file
            if len(self.replay_buffer) >= self.params.batch_size:
                print("Running bootstrap training iteration...")
                self.train_iteration(is_replay_buffer_bootstrap=True)
                self.first_replay_buffer_loaded = True

    def version(self):
        return "1.0"

    def model_name(self):
        return "alphazero"

    def is_training(self):
        return self.params.training_mode

    @classmethod
    def params_class(cls):
        return AlphaZeroParams

    def model_id(self):
        return f"{self.model_name()}_B{self.board_size}W{self.max_walls}_mv{self.version()}"

    def resolve_filename(self, suffix):
        return f"{self.model_id()}{suffix}.pt"

    def wandb_local_filename(self, artifact: wandb.Artifact) -> str:
        return f"{self.model_id()}_{artifact.digest[:5]}.pt"

    def get_model_state(self) -> dict:
        # Save the neural network state dict
        nn = self.evaluator.network
        model_state = {
            "network_state_dict": nn.state_dict(),
            "episode_count": self.episode_count,
            "board_size": self.board_size,
            "max_walls": self.max_walls,
            "params": self.params.__dict__,
        }
        return model_state

    def save_model(self, path):
        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)
        model_state = self.get_model_state()
        torch.save(model_state, path)
        print(f"AlphaZero model saved to {path}")

    def set_model_state(self, model_state: dict):
        self.evaluator.network.load_state_dict(model_state["network_state_dict"])

    def load_model(self, path):
        model_state = torch.load(path, map_location=my_device())
        self.set_model_state(model_state)

    def start_game(self, game, player_id):
        self.visited_states.clear()

    def end_game(self, env):
        if not self.params.training_mode:
            return

        # Assign the final game outcome to all positions in this episode
        # For Quoridor: reward = 1 for win, -1 for loss, 0 for draw
        episode_positions = []
        while self.replay_buffer and self.replay_buffer[-1]["value"] is None:
            position = self.replay_buffer.pop()
            agent = env.player_to_agent[position["player"]]
            position["value"] = env.rewards[agent]
            episode_positions.append(position)

        # Add back the positions with assigned values
        self.replay_buffer.extend(reversed(episode_positions))

        self.episode_count += 1

        # Train network if we have enough episodes
        if self.params.train_every is not None and self.episode_count % self.params.train_every == 0:
            self.train_iteration()

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # Return some basic metrics if available
        if hasattr(self, "recent_losses") and self.recent_losses:
            avg_loss = np.mean(self.recent_losses[-length:])
        else:
            avg_loss = 0.0

        return float(avg_loss), 0.0

    def train_iteration(self, is_replay_buffer_bootstrap=False):
        """Train the neural network on collected data."""
        if len(self.replay_buffer) < self.params.batch_size:
            return

        # Save replay buffer if requested
        if not is_replay_buffer_bootstrap:
            if self.params.save_replay_buffer == "always" or (
                self.params.save_replay_buffer == "first"
                and not self.first_replay_buffer_saved
                and not self.first_replay_buffer_loaded
            ):
                self.save_replay_buffer_to_file(self.episode_count)
                self.first_replay_buffer_saved = True

        t0 = time.time()
        print(f"Training the network (buffer size: {len(self.replay_buffer)}, batch size: {self.params.batch_size})...")

        metrics = self.evaluator.train_iteration(self.replay_buffer, self.params.validation_ratio)

        # Store loss for metrics
        self.recent_losses.append(metrics["total_loss"])
        if len(self.recent_losses) > 100:  # Keep only recent losses
            self.recent_losses = self.recent_losses[-100:]

        print(f"Finished training in {time.time() - t0:.2f}s")

    def _replay_buffer_filename(self, episode_number: int):
        params = {
            "ep": episode_number,
            "i": get_initial_random_seed(),
            "t": int(self.params.temperature * 100),
            "rbs": self.params.replay_buffer_size,
            "n": self.params.mcts_n,
            "k": self.params.mcts_k,
            "ucbc": int(self.params.mcts_ucb_c * 100),
            "pvs": "" if self.params.penalized_visited_states else None,
            "frbl": "" if self.first_replay_buffer_loaded else None,
        }

        if self.first_replay_buffer_saved:
            # If we already saved the first replay buffer, it means that we already run a training batch,
            # in which case we want to include the training parameters so we don't override a file with
            # different parameters (those files are useful for visualization).
            # On the other hand, if it's for the first replay buffer, we don't want to include the parameters
            # below because the replay buffer doesn't depend on them, and if we did, we wouldn't be able to load it
            # when training with different parameters.
            params = params | {
                "lr": int(self.params.learning_rate * 1000000),
                "bs": self.params.batch_size,
                "oi": self.params.optimizer_iterations,
            }
        filtered = filter(lambda x: x[1] is not None, params.items())
        str_params = "_".join(f"{x[0]}{x[1]}" for x in filtered)
        return f"alphazero_B{self.board_size}W{self.max_walls}_{str_params}.pkl"

    def save_replay_buffer_to_file(self, episode_number: int):
        """Save replay buffer contents to a file."""
        replay_buffer_dir = "replay_buffers"
        os.makedirs(replay_buffer_dir, exist_ok=True)
        filepath = os.path.join(replay_buffer_dir, self._replay_buffer_filename(episode_number))
        with open(filepath, "wb") as f:
            pickle.dump(list(self.replay_buffer), f)
        print(f"Saved replay buffer to {filepath}")

    def load_replay_buffer_from_file(self) -> bool:
        """Load replay buffer from file if it exists. Returns True if the file exists."""
        filepath = os.path.join("replay_buffers", self._replay_buffer_filename(self.params.train_every))
        if not os.path.exists(filepath):
            return False

        with open(filepath, "rb") as f:
            bootstrap_data = pickle.load(f)
        self.replay_buffer.extend(bootstrap_data)
        print(f"Loaded {len(bootstrap_data)} samples from {filepath}")
        return True

    def _log_action(
        self,
        visit_probs,
        root_children,
        root_value,
        root_action,
    ):
        if not self.action_log.is_enabled():
            return

        self.action_log.clear()

        _, top_indices = torch.topk(torch.tensor(visit_probs), min(5, len(visit_probs)))

        scores = {root_children[i].action_taken: visit_probs[i] for i in top_indices}
        self.action_log.action_score_ranking(scores)
        self.action_log.action_text(root_action, f"{root_value:0.2f}")

    def get_action(self, observation) -> int:
        game, player, _ = construct_game_from_observation(observation["observation"])
        # Run MCTS to get action visit counts
        root_children, root_value = self.mcts.search(game)
        visit_counts = np.array([child.visit_count for child in root_children])
        visit_counts_sum = np.sum(visit_counts)
        if visit_counts_sum == 0:
            raise RuntimeError("No nodes visited during MCTS")

        visit_probs = visit_counts / visit_counts_sum
        self._log_action(
            visit_probs, root_children, float(root_value), MoveAction(game.board.get_player_position(player))
        )

        if self.temperature == 0.0:
            max_value = np.max(visit_probs)
            visit_probs = np.array([1.0 if v == max_value else 0.0 for v in visit_probs])
            visit_probs /= np.sum(visit_probs)
        else:
            visit_probs = visit_probs ** (1.0 / self.temperature)
            visit_probs = visit_probs / np.sum(visit_probs)

        # Sample from probability distribution
        best_child = np.random.choice(root_children, p=visit_probs)
        action = best_child.action_taken

        if self.params.penalized_visited_states:
            self.visited_states.add(QuoridorKey(best_child.game))
        # Store training data if in training mode
        if self.params.training_mode:
            # Convert visit counts to policy target (normalized)
            policy_target = self.action_encoder.get_action_mask_template()
            for child in root_children:
                action_index = self.action_encoder.action_to_index(child.action_taken)
                policy_target[action_index] = child.visit_count / visit_counts_sum
            self.store_training_data(game, policy_target, player)

        return self.action_encoder.action_to_index(action)

    def store_training_data(self, game, mcts_policy, player):
        """Store training data for later use in training."""
        game, is_rotated = self.evaluator.rotate_if_needed_to_point_downwards(game)
        input_array = self.evaluator.game_to_input_array(game)
        if is_rotated:
            mcts_policy = self.evaluator.rotate_policy_from_original(mcts_policy)
        self.replay_buffer.append(
            {
                "input_array": input_array,
                "mcts_policy": mcts_policy,
                "value": None,  # Will be filled in at end of episode
                "player": player,
            }
        )
