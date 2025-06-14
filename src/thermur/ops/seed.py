"""
Utilities for ensuring reproducible results via random seeding.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries.

    This function seeds `random`, `numpy`, and `torch` to ensure that any
    stochastic processes in the application are repeatable.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # The following are needed for full determinism with CUDA
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    logger.info(f"Global random seed set to {seed}.")
