from typing import Dict, Type

import numpy as np
import torch.nn as nn
from gymnasium import Space, spaces


class BaseNN(nn.Module):
    _registry: Dict[str, Type["BaseNN"]] = {}

    def __init__(self):
        super().__init__()

    @classmethod
    def id(cls) -> str:
        """Return unique identifier for the network architecture."""
        return cls.__name__

    def __init_subclass__(cls, **kwargs):
        """Automatically register child classes."""
        super().__init_subclass__(**kwargs)
        BaseNN._registry[cls.id()] = cls

    @classmethod
    def get_network(cls, network_id: str) -> Type["BaseNN"]:
        """Get network class by ID."""
        if network_id not in cls._registry:
            raise ValueError(f"Network '{network_id}' not found in registry")
        return cls._registry[network_id]

    def observation_to_tensor(self, observation):
        """Convert the observation dict to a tuple with tensors used as input of the NN."""
        raise NotImplementedError("observation to tensor needs to be implemented by the NN")

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

    def _calculate_action_size(self, action_space):
        """Calculate the size of the action space."""
        return action_space.n if isinstance(action_space, spaces.Discrete) else self._get_space_size(action_space)

    def _calculate_observation_size(self, observation_space):
        """Calculate the size of the action space."""
        return self._get_space_size(observation_space["observation"])
