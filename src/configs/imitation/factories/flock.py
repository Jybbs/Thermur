"""
Hydra-zen builders for flock data specifications.

This module provides factory functions that create torchrl spec objects
through builder functions that are compatible with Hydra's serialization.
"""
from ..schemas    import FlockModel
from hydra_zen    import builds
from omegaconf    import SI
from torchrl.data import Composite, Unbounded, Bounded

import torch


def create_edge_index_spec(n: int):
    """
    Creates the edge index spec for dynamic graph connectivity.
    
    For n agents, defines edges for the communication graph G_t = (V, E_t).
    Shape [2, n*(n-1)] encodes source and target nodes for all possible edges
    in a directed graph, enabling metric-based neighborhood computation.
    
    Args:
        n: Number of agents in the flock
        
    Returns:
        TensorSpec for graph edge indices with int64 dtype
    """
    return Bounded(
        low    = 0,
        high   = n - 1,
        shape  = (2, n * (n - 1)),
        device = "cpu",
        dtype  = torch.int64,
    )


def create_float_spec(shape):
    """
    Helper to create float32 TensorSpec with configurable shape.
    """
    return Unbounded(
        shape = shape,
        dtype = torch.float32
    )


def create_action_spec(agent_count: int, spatial_dims: int):
    """
    Creates the action specification for flock control.
    
    Defines the action space as nominal velocity commands u_nom ∈ ℝᵈ for each
    agent. These commands are processed by the safety filter before execution
    to ensure thermal constraint satisfaction.
    
    Args:
        agent_count: Number of agents N to control
        spatial_dims: Dimensionality d of velocity commands
        
    Returns:
        Composite spec for agent actions
    """
    return Composite(
        action = create_float_spec((agent_count, spatial_dims)),
    )


def create_observation_spec(agent_count: int, spatial_dims: int):
    """
    Creates the observation specification for thermal flock state.
    
    Defines the complete observation space for each agent:
        - battery          : Energy remaining E ∈ [0,1]
        - edge_index       : Dynamic graph topology for inter-agent communication
        - position         : Spatial coordinates x ∈ ℝᵈ
        - temperature      : Thermal state T ∈ ℝ
        - temperature_grad : Thermal gradient ∇T ∈ ℝᵈ
        - velocity         : Motion vectors v ∈ ℝᵈ  
        - wind             : Wind velocity vector w ∈ ℝᵈ at agent position
        
    Args:
        agent_count  : Number of agents N in the flock
        spatial_dims : Dimensionality d of the spatial environment (2D/3D)
        
    Returns:
        Composite spec defining the full observation structure
    """
    return Composite(
        battery          = create_float_spec((agent_count, 1)),
        edge_index       = create_edge_index_spec(agent_count),
        position         = create_float_spec((agent_count, spatial_dims)),
        temperature      = create_float_spec((agent_count, 1)),
        temperature_grad = create_float_spec((agent_count, spatial_dims)),
        velocity         = create_float_spec((agent_count, spatial_dims)),
        wind             = create_float_spec((agent_count, spatial_dims)),
    )


# Builders that use structured interpolation
build_action_spec = builds(
    create_action_spec,
    agent_count  = SI("${flock.agent_count}"),
    spatial_dims = SI("${flock.spatial_dims}"),
)
"""
Builder for the action space specification.

Defines the continuous velocity control space for each agent in the flock,
compatible with TorchRL's environment interface.
"""

build_observation_spec = builds(
    create_observation_spec,
    agent_count  = SI("${flock.agent_count}"),
    spatial_dims = SI("${flock.spatial_dims}"),
)
"""
Builder for the observation space specification.

Defines the state space including agent kinematics, communication graph
structure, and environmental temperature data for policy inputs.
"""

build_flock = builds(
    FlockModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.configs.imitation.factories.flock",
        "cls_name" : "FlockBuild"
    }
)
"""
Builder for flock configuration.

Creates a Pydantic-validated flock configuration that defines agent properties
including count, spatial dimensions, and temperature constraints.
"""