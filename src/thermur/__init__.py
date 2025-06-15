"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
__version__ = "0.1.0"

from .cli        import app
from .models     import GNNPolicy
from .simulation import SwarmData, SwarmDataSpec, SimulationEnv, compute_edge_index
from .control    import ExpertFlockingController
from .utils      import EnvironmentDataSource, configure_loguru, set_seed
from .training   import train_imitation_learning, save_checkpoint, ImitationLoss

__all__ = [
    # Version
    "__version__",

    # Simulation components
    "SwarmData", 
    "SwarmDataSpec",
    "SimulationEnv",
    "compute_edge_index",
    
    # CLI components
    "app",

    # Models
    "GNNPolicy",

    # Utilities
    "EnvironmentDataSource",
    "configure_loguru",
    "set_seed",

    # Control components
    "ExpertFlockingController",

    # Training
    "train_imitation_learning",
    "save_checkpoint",
    "ImitationLoss",
]
