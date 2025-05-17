import json
import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    @classmethod
    def _to_storage_format(cls, state):
        if isinstance(state, torch.Tensor):
            return state.cpu().numpy()
        elif isinstance(state, list):
            return [ReplayBuffer._to_storage_format(item) for item in state]
        elif isinstance(state, tuple):
            return tuple(ReplayBuffer._to_storage_format(item) for item in state)
        else:
            return state

    @classmethod
    def _from_storage_format(cls, state):
        if isinstance(state, np.ndarray):
            return torch.from_numpy(state)
        elif isinstance(state, tuple):
            return tuple(ReplayBuffer._from_storage_format(item) for item in state)
        elif isinstance(state, list):
            return [ReplayBuffer._from_storage_format(item) for item in state]
        elif isinstance(state, (int, float, bool, str)):
            return state
        else:
            raise ValueError(f"Unexpected state type {type(state)}")

    def add(self, state, action, reward, next_state, done, next_state_mask):
        self.buffer.append(
            [
                ReplayBuffer._to_storage_format(state),
                action,
                reward,
                ReplayBuffer._to_storage_format(next_state),
                done,
                ReplayBuffer._to_storage_format(next_state_mask),
            ]
        )

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        samples = random.sample(self.buffer, batch_size)
        return [
            [
                ReplayBuffer._from_storage_format(state),
                action,
                reward,
                ReplayBuffer._from_storage_format(next_state),
                done,
                ReplayBuffer._from_storage_format(next_state_mask),
            ]
            for state, action, reward, next_state, done, next_state_mask in samples
        ]

    def get_last(self):
        return self.buffer[-1]

    def __len__(self):
        return len(self.buffer)

    def to_disk(self, filename):
        serializable_buffer = [
            [
                state.tolist() if hasattr(state, "tolist") else state,
                action,
                reward,
                next_state.tolist() if hasattr(next_state, "tolist") else next_state,
                done,
                next_state_mask.tolist() if hasattr(next_state_mask, "tolist") else next_state_mask,
            ]
            for state, action, reward, next_state, done, next_state_mask in self.buffer
        ]
        with open(filename, "w") as f:
            json.dump(serializable_buffer, f)
