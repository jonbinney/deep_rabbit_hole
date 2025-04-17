import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, next_state_mask):
        self.buffer.append([state, action, reward, next_state, done, next_state_mask])
        if reward >= 10:
            self.buffer.append([state, action, reward, next_state, done, next_state_mask])
            self.buffer.append([state, action, reward, next_state, done, next_state_mask])
            self.buffer.append([state, action, reward, next_state, done, next_state_mask])
            self.buffer.append([state, action, reward, next_state, done, next_state_mask])

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_last(self):
        return self.buffer[-1]

    def __len__(self):
        return len(self.buffer)
