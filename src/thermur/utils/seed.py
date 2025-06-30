"""
Utilities for ensuring reproducible results via random seeding.
"""
from __future__ import annotations
from loguru     import logger
from numpy      import random
from torch      import backends, cuda, manual_seed


def set_seed(seed: int):
    """
    Sets the random seed for all relevant libraries.

    This function seeds `random`, `numpy`, and `torch` to ensure that any
    stochastic processes in the application are repeatable.

    Args:
        seed: The integer seed to use.
    """
    random.seed(seed)
    manual_seed(seed)
    
    if cuda.is_available():
        cuda.manual_seed(seed)
        cuda.manual_seed_all(seed)

        # The following are needed for full determinism with CUDA
        backends.cudnn.deterministic = True
        backends.cudnn.benchmark     = False

    logger.info(f"Global random seed set to {seed}.")
