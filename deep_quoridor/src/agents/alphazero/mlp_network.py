import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    def __init__(self, input_size, action_size, device):
        super(MLPNetwork, self).__init__()

        self.device = device

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head - outputs action probabilities
        # TODO: Is it correct to include the Softmax at the end? Some implementations of alphazero
        # appear to leave it out, or apply it outside the network.
        self.policy_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, action_size), nn.Softmax(dim=0))

        # value head - outputs position evaluation (-1 to 1)
        self.value_head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

        self.to(self.device)

    def forward(self, x):
        if isinstance(x, tuple):
            x = torch.stack([i for i in x]).to(self.device)

        shared_features = self.shared(x)
        policy = self.policy_head(shared_features)
        value = self.value_head(shared_features)

        return policy, value
