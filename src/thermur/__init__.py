"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
from .cli        import app
from .control    import ExpertFlockingController
from .models     import GNNPolicy
from .simulation import SwarmData, SwarmDataSpec, SimulationEnv, compute_edge_index
from .training   import train_imitation_learning, save_checkpoint, ImitationLoss
from .utils      import EnvironmentDataSource, configure_loguru, set_seed

__all__ = [

    # CLI components
    "app",

    # Control components
    "ExpertFlockingController",

    # Models
    "GNNPolicy",

    # Simulation components
    "SwarmData", 
    "SwarmDataSpec",
    "SimulationEnv",
    "compute_edge_index",

    # Training
    "train_imitation_learning",
    "save_checkpoint",
    "ImitationLoss",

    # Utilities
    "EnvironmentDataSource",
    "configure_loguru",
    "set_seed",

]
