import json
import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_state_mask):
        self.buffer.append([state, action, reward, next_state, done, next_state_mask])

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

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
