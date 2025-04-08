import random
from pathlib import Path
from typing import Optional

import gymnasium.utils.seeding
import numpy as np
import torch


def resolve_path(dir: str, filename: Optional[str] = None) -> Path:
    path = Path(dir)
    if not path.is_absolute():
        # Update this if this file is moved
        path = Path(__file__).resolve().parents[3] / path

    return path / filename if filename else path


def set_deterministic(seed=42):
    """Sets all random seeds for deterministic behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gymnasium.utils.seeding.np_random(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
