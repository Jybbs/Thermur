"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
from __future__  import annotations
from .control    import ExpertFlockingController, SafetyFilter, ThermalBarrierFunction
from .models     import GNNPolicy
from .simulation import SimulationEnv, compute_edge_index
from .training   import train_imitation_learning, save_checkpoint, ImitationLoss
from .utils      import *

__all__ = [

    # Control components
    "ExpertFlockingController",
    "SafetyFilter",
    "ThermalBarrierFunction",

    # Models
    "GNNPolicy",

    # Simulation components
    "SimulationEnv",
    "compute_edge_index",

    # Training
    "train_imitation_learning",
    "save_checkpoint",
    "ImitationLoss",

    # Utilities
    "EnvironmentDataSource",
    "configure_loguru",
    "generate_swarm_xml",
    "load_swarm_model",
    "set_seed",

]
