import numpy as np
import torch
import torch.nn as nn

from agents.core import AbstractTrainableAgent, agent_utils


class DExpNetwork(nn.Module):
    """
    Neural network model for Deep Q-learning.
    Takes observation from the Quoridor game and outputs Q-values for each action.
    """

    def __init__(self, board_size, action_size, split_board, include_turn):
        super(DExpNetwork, self).__init__()

        # Calculate input dimensions based on observation space
        # Board is board_size x board_size with 2 channels (player position and opponent position)
        # Walls are (board_size-1) x (board_size-1) with 2 channels (vertical and horizontal walls)
        board_input_size = board_size * board_size * (2 if split_board else 1)
        walls_input_size = (board_size - 1) * (board_size - 1) * 2

        # Additional features: walls remaining for both players
        # turn, board player, board opponent, wall positions, my remaining walls, opponent's remaining walls
        flat_input_size = board_input_size + walls_input_size + (3 if include_turn else 2)

        # Define network architecture
        self.model = nn.Sequential(
            nn.Linear(flat_input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_size),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

    def forward(self, x):
        return self.model(x)


class DExpPlayParams:
    def __init__(self, use_rotate_board=False, include_turn=False, split_board=False):
        self.use_rotate_board = use_rotate_board
        self.include_turn = include_turn
        self.split_board = split_board

    @classmethod
    def from_str(cls, agent_params_str="000"):
        p = DExpPlayParams(
            bool(int(agent_params_str[0])), bool(int(agent_params_str[1])), bool(int(agent_params_str[2]))
        )
        return p

    def __str__(self):
        return f"{int(self.use_rotate_board)}{int(self.include_turn)}{int(self.split_board)}"


class DExpAgent(AbstractTrainableAgent):
    """Diego experimental Agent using DRL."""

    def __init__(
        self,
        use_opponentns_actions=False,
        params=DExpPlayParams(),
        agent_params_str=None,
        **kwargs,
    ):
        self.use_opponents_actions = use_opponentns_actions
        self.params = params
        if agent_params_str is not None:
            self.params = DExpPlayParams.from_str(agent_params_str)
        super().__init__(**kwargs)

    def resolve_filename(self, suffix):
        filename = f"dexp_B{self.board_size}W{self.max_walls}_C{self.params}_{suffix}.pt"
        return filename

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return self.board_size**2 + (self.board_size - 1) ** 2 * 2

    def _create_network(self):
        """Create the neural network model."""
        return DExpNetwork(self.board_size, self.action_size, self.params.split_board, self.params.include_turn)

    def handle_opponent_step_outcome(self, observation_before_action, action, game):
        if not self.training_mode or not self.use_opponents_actions:
            return

        opponent_player = "player_1" if self.player_id == "player_0" else "player_0"

        reward = game.rewards[opponent_player]

        # Handle end of episode
        if action is None:
            # TODO: Check whether it is worth it
            # if self.assign_negative_reward and len(self.replay_buffer) > 0:
            #    last = self.replay_buffer.get_last()
            #    last[2] = reward  # update final reward
            #    last[4] = 1.0  # mark as done
            #    self.current_episode_reward += reward
            return

        state_before_action = self.observation_to_tensor_by_player(observation_before_action, True, opponent_player)
        state_after_action = self.observation_to_tensor_by_player(game.observe(opponent_player), False, opponent_player)
        done = game.is_done()

        self.replay_buffer.add(
            state_before_action.cpu().numpy(),
            action,
            reward,
            state_after_action.cpu().numpy()
            if state_after_action is not None
            else np.zeros_like(state_before_action.cpu().numpy()),
            float(done),
        )

        if len(self.replay_buffer) > self.batch_size:
            loss = self.train(self.batch_size)
            if loss is not None:
                self.train_call_losses.append(loss)

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a flat tensor."""
        obs = observation["observation"]
        my_turn = 1 if obs["my_turn"] else 0
        return self.observation_to_tensor_by_player(observation, my_turn, self.player_id)

    def observation_to_tensor_by_player(self, observation, my_turn, player_id):
        """Convert the observation dict to a flat tensor."""
        obs = observation["observation"]

        board = (
            obs["board"]
            if (player_id == "player_0" or not self.params.use_rotate_board)
            else agent_utils.rotate_board(obs["board"])
        )
        walls = (
            obs["walls"]
            if (player_id == "player_0" or not self.params.use_rotate_board)
            else agent_utils.rotate_walls(obs["walls"])
        )

        # Split the board into player and opponent positions
        player_board = (board == 1 if my_turn else board == 2).astype(np.float32)
        opponent_board = (board == 2 if my_turn else board == 1).astype(np.float32)
        my_walls = np.array([obs["my_walls_remaining"]])
        opponent_walls = np.array([obs["opponent_walls_remaining"]])
        if not my_turn and self.params.include_turn:
            my_walls, opponent_walls = opponent_walls, my_walls
            player_board, opponent_board = opponent_board, player_board
        board = np.stack([player_board, opponent_board]) if self.params.split_board else board
        board = board.flatten()
        walls = walls.flatten()
        turn = [my_turn] if self.params.include_turn else []
        flat_obs = np.concatenate([turn, board, walls, my_walls, opponent_walls])
        return torch.FloatTensor(flat_obs).to(self.device)

    def convert_action_mask_to_tensor(self, mask):
        """Convert action mask to tensor, rotating it for player_1."""
        if self.player_id == "player_0" or not self.params.use_rotate_board:
            return torch.tensor(mask, dtype=torch.float32, device=self.device)
        rotated_mask = agent_utils.rotate_action_mask(self.board_size, mask)
        return torch.tensor(rotated_mask, dtype=torch.float32, device=self.device)

    def convert_to_action_from_tensor_index(self, action_index_in_tensor):
        """Convert action index from rotated tensor back to original action space."""
        if self.player_id == "player_0" or not self.params.use_rotate_board:
            return super().convert_to_action_from_tensor_index(action_index_in_tensor)

        return agent_utils.convert_rotated_action_index_to_original(self.board_size, action_index_in_tensor)


class DExpPretrainedAgent(DExpAgent):
    """
    A DExpAgent that is initialized with the pre-trained model from main.py.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load_pretrained_file()
