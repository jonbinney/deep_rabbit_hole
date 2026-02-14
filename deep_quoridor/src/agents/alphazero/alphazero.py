import os
import pickle
import random
import shutil
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import wandb
import wandb.wandb_run
from agents.alphazero.mcts import MCTS, QuoridorKey
from agents.alphazero.nn_evaluator import NNConfig, NNEvaluator
from agents.core import TrainableAgent
from pydantic_yaml import to_yaml_file
from quoridor import ActionEncoder, MoveAction, construct_game_from_observation
from utils import Timer, get_initial_random_seed, my_device, resolve_path
from utils.subargs import SubargsBase


@dataclass
class AlphaZeroParams(SubargsBase):
    training_mode: bool = False

    # Just used to display a user friendly name
    nick: Optional[str] = None

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
    # If not set, it will use 1 for training and 0 for playing (unless drop_t_on_step is set,
    # in which case it will be set to 1)
    temperature: Optional[float] = None

    # If set, on that number of moves and afterwards the temperature will be set to 0
    drop_t_on_step: Optional[int] = None

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

    # If wandb_alias is provided, the model will be fetched from wandb using the model_id and the alias
    wandb_alias: Optional[str] = None

    # When loading from wandb, the project name to be used
    wandb_project: str = "deep_quoridor"

    # If a filename is provided, the model will be loaded from disc
    model_filename: Optional[str] = None

    # Directory where wandb models are stored
    wandb_dir: str = "wandbmodels"

    # Directory where local models are stored
    model_dir: str = "models"

    # If True, the agent will penalize visited states in MCTS to avoid cycling
    penalized_visited_states: bool = False

    # Whether to save the replay buffer to a file.  This can be used to fast-forward the first batch of games
    # (in which case you can only use the first replay buffer) or to use a tool to visualize any of the replay buffers.
    # The options are: "never" | "first" | "always"
    save_replay_buffer: str = "never"

    # Allows to deduplicate the new entries in the replay buffer after each epoch
    # - "none": doesn't de-duplicate the entries
    # - "log": the number of appearances is reduced to its log2 quantity.
    #          As repeated entries appear, only the ones in position 1, 2, 4, 8, 16 are kept
    replay_buffer_dedup: str = "none"

    # If non-zero, the reward will have a bonus factor based on the game length, making it
    # better to win in a shorter game than to win in a longer game, and better to lose in a
    # longer game than a shorter one.
    game_length_bonus_factor: float = 0.0

    # What fraction of the total moves will be used for validation during training, or 0.0 to not use a validation set
    validation_ratio: float = 0.0

    # What fraction of the total moves will be used for the test set during training, or 0.0 to not use a test set.
    # Unlike the validation set, the test set remains the same throughout the entire run - it is created at the start
    # of the run and those states are never used for training.
    test_ratio: float = 0.0

    # What neural network structure to use
    # The options are "mlp" or "resnet"
    nn_type: str = "mlp"

    # Number of residual blocks in the resnet. Only used if nn_type is set to "resnet"
    # If set to None, then 2*(dimension of combined grid)+2 is used. Alphazero used a value of 20 for chess
    # and 40 for Go, so we also choose something a little more than double the input dimension.
    nn_resnet_num_blocks: Optional[int] = None

    # Number of channels used internally between residual blocks. Only used if nn_type is set to "resnet"
    # Alphazero used 256. It's set lower here to make training faster, but we should try a higher value.
    nn_resnet_num_channels: int = 32

    # Whether to mask the policies predicted by the NN during training (before computing the loss). When this is
    # False, the loss function penalizes the network producing a non-zero probability for any action which is
    # illegal.
    nn_mask_training_predictions: bool = False

    # Maximum size of for entries in worker cache
    max_cache_size: int = 200000

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


@dataclass
class AlphaZeroBenchmarkOverrideParams(SubargsBase):
    temperature: Optional[float] = None
    drop_t_on_step: Optional[int] = None
    mcts_n: Optional[int] = None
    mcts_k: Optional[int] = None
    mcts_ucb_c: Optional[float] = None


class AlphaZeroAgent(TrainableAgent):
    def __init__(
        self,
        board_size: int,
        max_walls: int,
        max_steps: int = -1,  # -1 means games are never truncated.
        observation_space=None,
        action_space=None,
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

        nn_config = NNConfig.from_alphazero_params(params)
        self.evaluator = NNEvaluator(self.action_encoder, self.device, nn_config, params.max_cache_size)

        self._fetch_model_from_wandb_and_update_params()
        self._resolve_and_load_model()

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
        )

        if params.training_mode and params.train_every is not None:
            self.evaluator.train_prepare(
                params.learning_rate, params.batch_size, params.optimizer_iterations, params.weight_decay
            )

        self.episode_count = 0

        # When playing use 0.0 for temperature so we always chose the best available action.
        if params.temperature:
            self.initial_temperature = params.temperature
        else:
            if params.training_mode or params.drop_t_on_step is not None:
                self.initial_temperature = 1
            else:  # Play mode and params.drop_t_on_step is None
                self.initial_temperature = 0

        self.replay_buffer = deque(maxlen=params.replay_buffer_size)
        self.replay_buffers_in_progress = []

        # Metrics tracking
        self.recent_losses = []

        # Just to make the training status plugin happy
        self.epsilon = 0.0

        self.first_replay_buffer_saved = False
        self.first_replay_buffer_loaded = False

        if self.load_replay_buffer_from_file():
            # Run initial training if we have the saved file
            if len(self.replay_buffer) >= self.params.batch_size:
                print("Running bootstrap training iteration...")
                self.train_iteration(is_replay_buffer_bootstrap=True)
                self.first_replay_buffer_loaded = True

        self.wandb_run = None

        # Any game state whose hash has a least significant byte that is in this set is part of the
        # test set. The test set is the same for the entire run.
        self.test_set_lsbs = set(random.sample(range(256), int(round(256 * params.test_ratio))))

    def set_wandb_run(self, wandb_run: wandb.wandb_run.Run):
        self.wandb_run = wandb_run

    def wandb_artifact(self):
        alias = self.params.wandb_alias
        if not alias:
            return None

        api = wandb.Api()
        path = f"the-lazy-learning-lair/{self.params.wandb_project}/{self.model_id()}:{alias}"
        return api.artifact(path, type="model")

    # TO DO, this was copy-pasted from AbstractTrainableAgent, need to refactor it
    def _fetch_model_from_wandb_and_update_params(self):
        """
        This function doesn't do anything if wandb_alias is not set in self.params.
        Otherwise, it will download the file if there's not a local copy.
        The params are updated to the artifact metadata.

        """
        artifact = self.wandb_artifact()
        if artifact is None:
            return
        local_filename = resolve_path(self.params.wandb_dir, self.wandb_local_filename(artifact))

        self.params.model_filename = str(local_filename)

        if os.path.exists(local_filename):
            return local_filename

        print(f"{self.name()} - Fetching model from wandb: {artifact.name}")

        os.makedirs(local_filename.parent, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = artifact.download(root=tmpdir)

            # NOTE: This picks the first .pt file it finds in the artifact
            tmp_filename = next(Path(artifact_dir).glob(f"**/*.{self.get_model_extension()}"), None)
            if tmp_filename is None:
                raise FileNotFoundError(f"No model file found in artifact {artifact.name}")

            shutil.copyfile(tmp_filename, local_filename)

            print(f"Model downloaded from wandb to {local_filename}")

    def _resolve_and_load_model(self):
        """Figure out what model needs to be loaded based on the settings and loads it."""
        if self.params.model_filename:
            filename = self.params.model_filename
        else:
            # If no filename is passed in training mode, assume we are not loading a model
            if self.params.training_mode:
                return

            print("WARNING: no initial model provided usign a filename or wandb, so using random initialized model")
            return
            # If it's not training mode, we definitely need to load a pretrained model, so try the
            # default path for local files
            # filename = resolve_path(self.params.model_dir, self.resolve_filename("final"))

        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found.")

        self.load_model(filename)

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "alphazero"

    def version(self):
        return "1.0"

    def model_name(self):
        return "alphazero"

    def is_training(self):
        return self.params.training_mode

    @classmethod
    def get_model_extension(cls):
        return "pt"

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
            "test_set_lsbs": self.test_set_lsbs,
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

    def save_model_onnx(self, path):
        """Export the model to ONNX format."""
        import torch.onnx

        # Create directory for saving models if it doesn't exist
        os.makedirs(Path(path).absolute().parents[0], exist_ok=True)

        # Set the network to evaluation mode
        self.evaluator.network.eval()

        # Create a dummy input tensor with the correct shape
        # The shape depends on the network type
        network = self.evaluator.network
        if hasattr(network, "__class__") and network.__class__.__name__ == "ResnetNetwork":
            # ResNet expects input of shape (batch_size, 5, input_size, input_size)
            # NOTE: input_size is board_size * 2 + 3, which is the dimension of the combined grid input, not the original board size
            dummy_input = torch.randn(1, 5, network.input_size, network.input_size, device=self.device)
        else:
            # MLP expects input of shape (batch_size, input_size)
            dummy_input = torch.randn(1, network.input_size, device=self.device)

        # Export the model with opset 17 (widely supported, avoids version conversion issues)
        torch.onnx.export(
            self.evaluator.network,
            (dummy_input,),  # Args must be a tuple
            path,
            export_params=True,
            opset_version=17,  # Use 17 instead of 11 to avoid conversion issues
            do_constant_folding=True,
            input_names=["input"],
            output_names=["policy_logits", "value"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "policy_logits": {0: "batch_size"},
                "value": {0: "batch_size"},
            },
        )
        print(f"AlphaZero model exported to ONNX at {path}")

    def save_model_with_suffix(self, suffix: str) -> Path:
        path = resolve_path(self.params.model_dir, self.resolve_filename(suffix))
        self.save_model(path)
        return path

    def load_model(self, path):
        self.evaluator.load_model(path)

    def start_game_batch(self, envs):
        self.visited_states.clear()
        self.game_envs = envs
        for e in self.game_envs:
            e.reset()

        self.replay_buffers_in_progress = [[] for _ in range(len(envs))]

    def end_game_batch(self, env=None):
        if not self.params.training_mode:
            return

        if env:
            self.game_envs = [env]
        for i, env in enumerate(self.game_envs):
            # Assign the final game outcome to all positions in this episode
            # For Quoridor: reward = 1 for win, -1 for loss, 0 for draw
            episode_positions = []
            replay_buffer = self.replay_buffers_in_progress[i]
            while replay_buffer and replay_buffer[-1]["value"] is None:
                position = replay_buffer.pop()
                agent = env.player_to_agent[position["player"]]
                factor = 1 + self.params.game_length_bonus_factor * (1 - env.game.completed_steps / env.max_steps)
                position["value"] = factor * env.rewards[agent]
                episode_positions.append(position)

            # Add back the positions with assigned values
            self.replay_buffer.extend(reversed(episode_positions))

            self.episode_count += 1

    def end_game_batch_and_save_replay_buffers(self, temp_dir: Path, ready_dir: Path, model_version: int):
        if not self.params.training_mode:
            return

        from v2.yaml_models import GameInfo

        for i, env in enumerate(self.game_envs):
            # Assign the final game outcome to all positions in this episode
            # For Quoridor: reward = 1 for win, -1 for loss, 0 for draw
            processed_replay_buffer = []
            replay_buffer = self.replay_buffers_in_progress[i]
            while replay_buffer and replay_buffer[-1]["value"] is None:
                position = replay_buffer.pop()
                agent = env.player_to_agent[position["player"]]
                factor = 1 + self.params.game_length_bonus_factor * (1 - env.game.completed_steps / env.max_steps)
                position["value"] = factor * env.rewards[agent]
                processed_replay_buffer.append(position)

            processed_replay_buffer = list(reversed(processed_replay_buffer))

            filename = f"game_{int(time.time() * 1000)}_{i}_{os.getpid()}"

            # We first save the game info to the ready directory; the trainer will look for the pkl file
            # first, so there's no risk of partial reads or race conditions
            game_info = GameInfo(
                model_version=model_version, game_length=len(processed_replay_buffer), creator=str(os.getpid())
            )
            to_yaml_file(ready_dir / f"{filename}.yaml", game_info)

            # For the binary, we first save it in a temp directory and move it right away, to avoid partial reads
            pkl_filename = temp_dir / f"{filename}.pkl"
            with open(pkl_filename, "wb") as f:
                pickle.dump(processed_replay_buffer, f)

            pkl_filename.rename(ready_dir / pkl_filename.name)

    def start_game(self, game, player_id):
        self.visited_states.clear()
        self.replay_buffers_in_progress = [[]]

    def end_game(self, env):
        self.end_game_batch(env)

        # This is just used when we use the old train.py script
        if self.params.train_every is not None and self.episode_count % self.params.train_every == 0:
            self.train_iteration()

    def compute_loss_and_reward(self, length: int) -> Tuple[float, float]:
        # Return some basic metrics if available
        if hasattr(self, "recent_losses") and self.recent_losses:
            avg_loss = np.mean(self.recent_losses[-length:])
        else:
            avg_loss = 0.0

        return float(avg_loss), 0.0

    def train_iteration(self, is_replay_buffer_bootstrap=False, epoch=None, episode=None) -> bool:
        """
        Train the neural network on collected data.

        Returns:
            True if training was done, False if not enough data was available.
        """

        def log_loss_entry(entry):
            i = entry["step"]
            t, p, v = entry["loss_total"], entry["loss_policy"], entry["loss_value"]

            if "loss_total_val" in entry:
                vt, vp, vv = entry["loss_total_val"], entry["loss_policy_val"], entry["loss_value_val"]
                print(f"{i:>7}     {t:>7.3f} {vt:>7.3f}     {p:>7.3f} {vp:>7.3f}     {v:>7.3f} {vv:>7.3f}")
            else:
                print(f"{i:>7} {t:>7.3f} {p:>7.3f} {v:>7.3f}")

            if self.wandb_run:
                # Log this to take care of half of the epoch, so we can
                # see clearly the different phases of training
                entry["Loss step"] = epoch + entry["completion"] * 0.5
                del entry["step"]
                del entry["completion"]
                self.wandb_run.log(entry)

        if len(self.replay_buffer) * (1.0 - self.params.validation_ratio) < self.params.batch_size:
            return False

        # Save replay buffer if requested
        if not is_replay_buffer_bootstrap:
            if self.params.save_replay_buffer == "always" or (
                self.params.save_replay_buffer == "first"
                and not self.first_replay_buffer_saved
                and not self.first_replay_buffer_loaded
            ):
                self.save_replay_buffer_to_file(self.episode_count)
                self.first_replay_buffer_saved = True

        Timer.start("training")
        print(f"Training the network (buffer size: {len(self.replay_buffer)}, batch size: {self.params.batch_size})...")

        if self.params.validation_ratio > 0.0:
            print("==== Training & Validation Loss ====")
            print("  Epoch       Total   Val Total   Policy  Val Policy  Value   Val Value")
        else:
            print("==== Training Loss ====")
            print("  Epoch   Total   Policy  Value")

        self.evaluator.train_iteration(
            self.replay_buffer, self.params.validation_ratio, self.test_set_lsbs, on_new_entry=log_loss_entry
        )

        Timer.finish("training", episode)

        return True

    def _replay_buffer_filename(self, episode_number: int):
        params = {
            "ep": episode_number,
            "i": get_initial_random_seed(),
            "t": int(self.params.temperature * 100) if self.params.temperature else None,
            "dt": self.params.drop_t_on_step,
            "rbs": self.params.replay_buffer_size,
            "n": self.params.mcts_n,
            "k": self.params.mcts_k,
            "ucbc": int(self.params.mcts_ucb_c * 100),
            "pvs": "" if self.params.penalized_visited_states else None,
            "frbl": "" if self.first_replay_buffer_loaded else None,
            "ne": self.params.mcts_noise_epsilon,
            "na": self.params.mcts_noise_alpha,
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
        if self.params.train_every is None or not self.is_training():
            return False

        assert isinstance(self.params.train_every, int), "train_every must be set to load replay buffer from file"
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

    def get_action_batch(self, observations_with_ids: list[tuple[int, dict]]) -> list[tuple[int, int]]:
        """
        Get actions for multiple observations.  Each entry in observation_with_ids needs a distinct id that can be
        an arbitrary number. The same id will be returned with the action matching the observation.
        """
        if not observations_with_ids:
            return []

        games = []
        players = []
        game_indices = []
        for game_idx, observation in observations_with_ids:
            g, p, _ = construct_game_from_observation(observation["observation"])
            games.append(g)
            players.append(p)
            game_indices.append(game_idx)

        root_children_batch, root_value_batch = self.mcts.search_batch(games)

        actions = []
        for game_idx, root_children, root_value, game, player in zip(
            game_indices, root_children_batch, root_value_batch, games, players
        ):
            visit_counts = np.array([child.visit_count for child in root_children])
            visit_counts_sum = np.sum(visit_counts)
            if visit_counts_sum == 0:
                raise RuntimeError("No nodes visited during MCTS")

            visit_probs = visit_counts / visit_counts_sum

            # _log_action is used to display information about the action (e.g. probabilities of moves) in the GUI.
            # We allow that only when there's one game at the time, since we won't be playing multiple games and
            # displaying them
            if len(observations_with_ids) == 1:
                self._log_action(
                    visit_probs, root_children, float(root_value), MoveAction(game.board.get_player_position(player))
                )

            temperature = self.initial_temperature
            if self.params.drop_t_on_step is not None and game.completed_steps >= self.params.drop_t_on_step:
                temperature = 0

            if temperature == 0.0:
                max_value = np.max(visit_probs)
                visit_probs = np.array([1.0 if v == max_value else 0.0 for v in visit_probs])
                visit_probs /= np.sum(visit_probs)
            else:
                visit_probs = visit_probs ** (1.0 / temperature)
                visit_probs = visit_probs / np.sum(visit_probs)

            # Sample from probability distribution
            best_child = np.random.choice(root_children, p=visit_probs)
            action = best_child.action_taken
            actions.append((game_idx, self.action_encoder.action_to_index(action)))

            # TODO: this is disabled with multiple ids because we would need to keep multiple visited states sets,
            # one per game. Is it worth implementing?
            if len(observations_with_ids) == 1 and self.params.penalized_visited_states:
                self.visited_states.add(QuoridorKey(best_child.game))

            # Store training data if in training mode
            if self.params.training_mode:
                # Convert visit counts to policy target (normalized)
                policy_target = np.zeros(self.action_encoder.num_actions, dtype=np.float32)
                for child in root_children:
                    action_index = self.action_encoder.action_to_index(child.action_taken)
                    policy_target[action_index] = child.visit_count / visit_counts_sum
                self.store_training_data(game, policy_target, player, game_idx)

        return actions

    def get_action(self, observation) -> int:
        action_batch = self.get_action_batch([(0, observation)])
        return action_batch[0][1]

    def store_training_data(self, game, mcts_policy, player, game_idx):
        """Store training data for later use in training."""
        game, is_rotated = self.evaluator.rotate_if_needed_to_point_downwards(game)
        input_array = self.evaluator.game_to_input_array(game)
        action_mask = game.get_action_mask()
        if is_rotated:
            mcts_policy = self.evaluator.rotate_policy_from_original(mcts_policy)

        self.replay_buffers_in_progress[game_idx].append(
            {
                "input_array": input_array,
                "mcts_policy": mcts_policy,
                "action_mask": action_mask,
                "value": None,  # Will be filled in at end of episode
                "player": player,
            }
        )
