"""
Hydra-zen builders for swarm data specifications.

This module provides factory functions that create torchrl spec objects
through builder functions that are compatible with Hydra's serialization.
"""
from __future__   import annotations
from hydra_zen    import builds
from omegaconf    import SI
from torchrl.data import Composite, TensorSpec

import torch


def create_edge_index_spec(n: int):
    """
    Creates the edge index spec for dynamic graph connectivity.
    
    For n agents, defines edges for the communication graph G_t = (V, E_t).
    Shape [2, n*(n-1)] encodes source and target nodes for all possible edges
    in a directed graph, enabling metric-based neighborhood computation.
    
    Args:
        n: Number of agents in the swarm
        
    Returns:
        TensorSpec for graph edge indices with int64 dtype
    """
    return TensorSpec(
        shape  = (2, n * (n - 1)),
        device = "cpu",
        dtype  = torch.int64,
    ).to_owned_by(())


def create_float_spec(shape):
    """
    Helper to create float32 TensorSpec with configurable shape.
    """
    return TensorSpec(
        shape = shape,
        dtype = torch.float32
    )


def create_action_spec(agent_count: int, spatial_dims: int):
    """
    Creates the action specification for swarm control.
    
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
        shape  = agent_count,
        action = create_float_spec(spatial_dims),
    )


def create_observation_spec(agent_count: int, spatial_dims: int):
    """
    Creates the observation specification for thermal swarm state.
    
    Defines the complete observation space for each agent:
        - position         : Spatial coordinates x ∈ ℝᵈ
        - velocity         : Motion vectors v ∈ ℝᵈ  
        - temperature      : Thermal state T ∈ ℝ
        - temperature_grad : Thermal gradient ∇T ∈ ℝᵈ
        - battery          : Energy remaining E ∈ [0,1]
        - edge_index       : Dynamic graph topology for inter-agent communication
        
    Args:
        agent_count: Number of agents N in the swarm
        spatial_dims: Dimensionality d of the spatial environment (2D/3D)
        
    Returns:
        Composite spec defining the full observation structure
    """
    return Composite(
        shape            = agent_count,
        battery          = create_float_spec(1),
        edge_index       = create_edge_index_spec(agent_count),
        position         = create_float_spec(spatial_dims),
        temperature      = create_float_spec(1),
        temperature_grad = create_float_spec(spatial_dims),
        velocity         = create_float_spec(spatial_dims),
    )


# Builders that use structured interpolation
build_action_spec = builds(
    create_action_spec,
    agent_count  = SI("${swarm.agent_count}"),
    spatial_dims = SI("${swarm.spatial_dims}"),
)

build_observation_spec = builds(
    create_observation_spec,
    agent_count  = SI("${swarm.agent_count}"),
    spatial_dims = SI("${swarm.spatial_dims}"),
)