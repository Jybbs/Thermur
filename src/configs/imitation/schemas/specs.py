"""
Tensor specification models.

This module defines the data structures for swarm observations and actions,
providing the definitive shape and dtype specifications for all tensors
passed between the environment, policy, and replay buffer.
"""
from __future__ import annotations
from pydantic   import BaseModel, Field
from typing     import Literal


class SwarmActionModel(BaseModel, extra="forbid"):
    """
    Schema for swarm control actions.
    
    The action is the desired d-dimensional velocity vector 𝐮 for each
    agent that is output by the policy. This represents the control input
    that drives agent motion in the simulation.
    
    The action space is continuous and bounded by the physical constraints
    of the drone dynamics, though these bounds are enforced by the physics
    engine rather than the action spec itself.
    """
    dtype: Literal["float32"] = Field(
        default     = "float32",
        description = (
            "Data type for velocity commands, using float32 for GPU efficiency "
            "and precision."
        )
    )
    shape: tuple[int, int] = Field(
        description = (
            "Shape of action tensor (N, d) defining the "
            "batch and action dimensions."
        )
    )


class SwarmObservationModel(BaseModel, extra="forbid"):
    """
    Schema for swarm observation data.
    
    An observation consists of agent-specific state vectors and the shared
    graph topology (edge_index). This defines the structure of the agent
    state vector s, where:
    
        s = [𝐩, 𝐯, T, ∇T, E]
    
    Components:
    - 𝐩  (position)         : Agent's d-dimensional position in the environment
    - 𝐯  (velocity)         : Agent's d-dimensional velocity vector
    - T  (temperature)      : Scalar temperature at agent's position
    - ∇T (temperature_grad) : Temperature gradient vector for thermal navigation
    - E  (battery)          : Remaining battery percentage [0, 1]
    
    The edge_index defines the communication graph, determining which agents
    can share information. This is computed dynamically based on the
    communication range and agent positions.
    """
    battery_shape: tuple[int, int] = Field(
        default     = (10, 1),
        description = (
            "Shape (N, 1) for normalized battery levels ranging from "
            "0 (depleted) to 1 (full)."
        )
    )
    position_shape: tuple[int, int] = Field(
        description = (
            "Shape (N, d) for agent positions in world "
            "coordinates."
        )
    )
    temperature_grad_shape: tuple[int, int] = Field(
        description = (
            "Shape (N, d) for thermal gradient vectors "
            "used in navigation."
        )
    )
    temperature_shape: tuple[int, int] = Field(
        description = (
            "Shape (N, 1) for scalar temperature readings at each "
            "agent's location."
        )
    )
    velocity_shape: tuple[int, int] = Field(
        description = (
            "Shape (N, d) for agent velocity vectors in m/s."
        )
    )
    edge_index_dtype: Literal["int64"] = Field(
        default     = "int64",
        description = (
            "Data type for graph edge indices, using int64 to support large "
            "swarms."
        )
    )
    edge_index_shape: tuple[int, int] = Field(
        description = (
            "Shape (2, E) for COO-format edge indices where E ≤ N(N-1) "
            "is the number of edges in the communication graph."
        )
    )
    state_dtype: Literal["float32"] = Field(
        default     = "float32",
        description = (
            "Data type for all state tensors, balancing precision with memory "
            "efficiency."
        )
    )