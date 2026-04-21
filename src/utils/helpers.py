"""
Small utility helpers shared across the project.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch

DECIMAL_SIGNS = 5


def rnd(x: float) -> float:
    """Round *x* to ``DECIMAL_SIGNS`` decimal places."""
    return round(x, DECIMAL_SIGNS)


def get_start_end_index(a: int, b: int) -> Tuple[int, int]:
    """
    Return a random ``(start, end)`` pair in *[a, b)* with at least 10 steps.

    Used by evaluation and environment resets to sample random episode windows.
    """
    start_index = np.random.randint(a, b - 20)
    end_index = np.random.randint(start_index + 10, b)
    return start_index, end_index


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic behaviour (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
