"""
Imitation learning configuration domain.

This package contains all configuration-related code for imitation learning,
including schemas, factories, and workloads organized by functionality.
"""
from __future__           import annotations
from .                    import factories, schemas, workloads
from .factories           import *
from .schemas             import *
from .workloads.imitation import imitation_config, register_configs

__all__ = [
    # Submodules
    "factories",
    "schemas", 
    "workloads",
    
    # Key workload exports
    "imitation_config",
    "register_configs",
    
    # Commonly used factories
    "build_collector",
    "build_flocking_controller",
    "build_policy",
    "build_simulation",
    
    # Commonly used schemas
    "ControlModel",
    "LearningModel",
    "PhysicsModel",
    "SafetyModel", 
    "SwarmModel",
    "VisualizationModel",
]