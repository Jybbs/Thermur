"""
The Thermur package: Core business logic and functionality.

This package provides a clean API for accessing all major components of the
Thermur system, including the environment, policies, safety filters, and
training orchestration.
"""
from thermur.core    import *
from thermur.models  import GNNPolicy
from thermur.ops     import EnvironmentDataSource, configure_loguru, set_seed
from thermur.physics import ExpertFlockingController, SafetyFilter

__all__ = [
    # From core (using *)
    "ImitationLoss",
    "SwarmData", 
    "SwarmDataSpec",
    "ThermurEnv",
    "TrainingOrchestrator",
    "cli_main",
    "compute_edge_index",
    
    # From models
    "GNNPolicy",
    
    # From ops
    "EnvironmentDataSource",
    "configure_loguru",
    "set_seed",
    
    # From physics
    "ExpertFlockingController",
    "SafetyFilter",
]
