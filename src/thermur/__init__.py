"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system for imitation learning training.
"""
from .control.flocking         import ExpertFlockingController
from .control.safety           import SafetyFilter, ThermalBarrierFunction
from .models.gnn_policy        import GNNPolicy
from .simulation.environment   import SimulationEnv
from .simulation.geometry      import compute_edge_index
from .training.imitation       import train_imitation_learning, save_checkpoint
from .training.loss            import ImitationLoss
from .utils.data               import EnvironmentDataSource
from .utils.logging            import configure_loguru
from .utils.seed               import set_seed
from .utils.xml                import generate_swarm_xml, load_swarm_model
from .visualization.visualizer import Visualizer


__all__ = [
    "ExpertFlockingController",
    "SafetyFilter",
    "ThermalBarrierFunction",
    "GNNPolicy",
    "SimulationEnv",
    "compute_edge_index",
    "train_imitation_learning",
    "save_checkpoint",
    "ImitationLoss",
    "Visualizer",
    "EnvironmentDataSource",
    "configure_loguru",
    "generate_swarm_xml",
    "load_swarm_model",
    "set_seed",
]
