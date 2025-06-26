"""
This module defines the canonical data structures for the project, providing
the definitive source for the shape, dtype, and semantics of all data passed
between the environment, policy, and replay buffer.

The swarm's state is represented as a collection of per-agent vectors and a
shared communication graph topology. This design enables efficient parallel
processing while maintaining the relational structure needed for GNN policies.
"""
from pydantic import BaseModel, Field
from typing   import Literal


class SwarmActionModel(BaseModel, extra="forbid"):
    """
    Schema for swarm control actions.
    
    The action is the desired N-dimensional velocity vector (𝐮) for each
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
            "Shape of action tensor (agent_count, spatial_dims) defining the "
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
    - 𝐩  (position)         : Agent's 3D position in the environment
    - 𝐯  (velocity)         : Agent's 3D velocity vector
    - T  (temperature)      : Scalar temperature at agent's position
    - ∇T (temperature_grad) : Temperature gradient vector for thermal navigation
    - E  (battery)          : Remaining battery percentage [0, 1]
    
    The edge_index defines the communication graph, determining which agents
    can share information. This is computed dynamically based on the
    communication range and agent positions.
    """
    # Agent state components
    battery_shape: tuple[int, int] = Field(
        default     = (10, 1),
        description = (
            "Shape (agent_count, 1) for normalized battery levels ranging from "
            "0 (depleted) to 1 (full)."
        )
    )
    position_shape: tuple[int, int] = Field(
        description = (
            "Shape (agent_count, spatial_dims) for agent positions in world "
            "coordinates."
        )
    )
    temperature_grad_shape: tuple[int, int] = Field(
        description = (
            "Shape (agent_count, spatial_dims) for thermal gradient vectors "
            "used in navigation."
        )
    )
    temperature_shape: tuple[int, int] = Field(
        description = (
            "Shape (agent_count, 1) for scalar temperature readings at each "
            "agent's location."
        )
    )
    velocity_shape: tuple[int, int] = Field(
        description = (
            "Shape (agent_count, spatial_dims) for agent velocity vectors in m/s."
        )
    )
    
    # Graph topology
    edge_index_dtype: Literal["int64"] = Field(
        default     = "int64",
        description = (
            "Data type for graph edge indices, using int64 to support large "
            "swarms."
        )
    )
    edge_index_shape: tuple[int, int] = Field(
        description = (
            "Shape (2, max_edges) for COO-format edge indices where max_edges "
            "= N*(N-1) for a fully connected graph."
        )
    )
    
    # All float tensors use float32
    state_dtype: Literal["float32"] = Field(
        default     = "float32",
        description = (
            "Data type for all state tensors, balancing precision with memory "
            "efficiency."
        )
    )
