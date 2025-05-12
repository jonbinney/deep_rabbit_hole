import numpy as np
from gymnasium import Space, spaces
from utils.misc import get_opponent_player_id

from agents.core import AbstractTrainableAgent
from agents.core.trainable_agent import TrainableAgentParams
from agents.nn.cnn_v1 import CnnV1Network
from agents.nn.cnn_v2 import CnnV2Network
from agents.nn.flat_1024 import Flat1024Network
from agents.nn.pyramid_512_dropout import P512DropoutNetwork


class AdaptableAgent(AbstractTrainableAgent):
    """Agent that does core DQN behaviour but expect observations and transformations to be done by adapters of this agent."""

    def name(self):
        if self.params.nick:
            return self.params.nick
        return "PAgent"

    def model_name(self):
        return "pagent"

    @classmethod
    def params_class(cls):
        return TrainableAgentParams

    def version(self):
        """Bump this version when compatibility with saved models is broken"""
        return 1

    def _get_space_size(self, space: Space) -> int:
        if isinstance(space, spaces.Discrete):
            return 1
        elif isinstance(space, spaces.Dict) or isinstance(space, dict):
            size = 0
            for subspace in space.values():
                size += self._get_space_size(subspace)  # Recursively get size of subspaces
            return size
        else:
            return int(np.prod(space.shape))

    def _calculate_action_size(self):
        """Calculate the size of the action space."""
        return (
            self.action_space.n
            if isinstance(self.action_space, spaces.Discrete)
            else self._get_space_size(self.action_space)
        )

    def _calculate_observation_size(self):
        """Calculate the size of the action space."""
        return self._get_space_size(self.observation_space["observation"])

    def _create_network(self):
        """Create the neural network model."""
        if self.params.nn_version is not None:
            if self.params.nn_version == Flat1024Network.id():
                return Flat1024Network(self._calculate_observation_size(), self._calculate_action_size())
            elif self.params.nn_version == P512DropoutNetwork.id():
                return P512DropoutNetwork(self._calculate_observation_size(), self._calculate_action_size())
            elif self.params.nn_version == CnnV1Network.id():
                return CnnV1Network(self._calculate_observation_size(), self._calculate_action_size())
            elif self.params.nn_version == CnnV2Network.id():
                return CnnV2Network(self._calculate_observation_size(), self._calculate_action_size())
            else:
                raise RuntimeError(f"Unknown nn: {self.params.nn_version}")
            # Default network
        return Flat1024Network(self._calculate_observation_size(), self._calculate_action_size())

    def handle_opponent_step_outcome(
        self,
        opponent_observation_before_action,
        my_observation_after_opponent_action,
        opponent_observation_after_action,
        opponent_reward,
        opponent_action,
        done=False,
    ):
        if not self.training_mode:
            return

        self._handle_step_outcome_all(
            opponent_observation_before_action,
            my_observation_after_opponent_action,
            opponent_observation_after_action,
            opponent_reward,
            opponent_action,
            get_opponent_player_id(self.player_id),
            done,
        )

    def _observation_to_tensor(self, observation, obs_player_id):
        return self.online_network.observation_to_tensor(observation)
