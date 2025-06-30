"""
Imitation learning configuration domain.

This package contains all configuration-related code for imitation learning,
including schemas, factories, and workloads organized by functionality.
"""
# Import submodules for easy access
from . import factories, schemas, workloads

# Re-export key items from workloads
from .workloads.imitation import imitation_config, register_configs

# For convenience, re-export commonly used factories
from .factories import (
    build_collector,
    build_flocking_controller,
    build_policy,
    build_simulation,
)

# For convenience, re-export commonly used schemas
from .schemas import (
    AgentModel,
    EnvironmentModel,
    FlockingModel,
    GNNModel,
    HyperparameterModel,
    SwarmModel,
)

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
    "AgentModel",
    "EnvironmentModel",
    "FlockingModel", 
    "GNNModel",
    "HyperparameterModel",
    "SwarmModel",
]