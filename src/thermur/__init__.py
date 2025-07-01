"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
# Control components
from .control.flocking import ExpertFlockingController
from .control.safety import SafetyFilter, ThermalBarrierFunction

# Models
from .models.gnn_policy import GNNPolicy

# Simulation components  
from .simulation.environment import SimulationEnv
from .simulation.geometry import compute_edge_index

# Training components
from .training.imitation import train_imitation_learning, save_checkpoint
from .training.loss import ImitationLoss

# Visualization
from .visualization.visualizer import Visualizer

# Utilities
from .utils.data import EnvironmentDataSource
from .utils.logging import configure_loguru
from .utils.xml import generate_swarm_xml, load_swarm_model
from .utils.seed import set_seed

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
    
    # Visualization
    "Visualizer",

    # Utilities
    "EnvironmentDataSource",
    "configure_loguru",
    "generate_swarm_xml",
    "load_swarm_model",
    "set_seed",
]
