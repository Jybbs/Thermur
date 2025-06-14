"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
from .core     import *
from .models   import GNNPolicy
from .ops      import EnvironmentDataSource, configure_loguru, set_seed
from .physics  import ExpertFlockingController
from .training import train_imitation_learning, save_checkpoint, ImitationLoss

__all__ = [

    # Core components
    "SwarmData", 
    "SwarmDataSpec",
    "ThermurEnv",
    "cli_main",
    "compute_edge_index",

    # Models
    "GNNPolicy",

    # Operations
    "EnvironmentDataSource",
    "configure_loguru",
    "set_seed",

    # Physics
    "ExpertFlockingController",

    # Training
    "train_imitation_learning",
    "save_checkpoint",
    "ImitationLoss",

]
