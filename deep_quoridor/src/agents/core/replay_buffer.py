import random
from collections import deque


class ReplayBuffer:
    """Experience replay buffer to store and sample transitions."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def get_last(self):
        return self.buffer[-1]

    def __len__(self):
        return len(self.buffer)


class ReplayBxuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity // 2)
        self.positive_buffer = deque(maxlen=capacity // 2)

    def add(self, state, action, reward, next_state, done):
        if reward > 0.5:
            self.positive_buffer.append([state, action, reward, next_state, done])
        else:
            self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        b1 = random.sample(self.positive_buffer, batch_size // 3) if len(self.positive_buffer) > 0 else []
        b2 = random.sample(self.buffer, batch_size - len(b1)) if len(self.buffer) > 0 else []
        merged = b1 + b2
        random.shuffle(merged)
        return merged

    def get_last(self):
        return self.buffer[-1]

    def __len__(self):
        return len(self.buffer)
