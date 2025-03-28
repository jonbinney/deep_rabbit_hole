import random

import gymnasium.utils.seeding
import numpy as np
import torch


def set_deterministic(seed=42):
    """Sets all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gymnasium.utils.seeding.np_random(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
