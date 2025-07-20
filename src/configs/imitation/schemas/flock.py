"""
Flock configuration model.

This module defines the unified configuration for the multi-agent flock,
including physical properties, collective behavior, and spatial settings.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


class TensorSpec(BaseModel, extra="forbid"):
    """
    Specification for tensor shape and type information.
    
    This lightweight model captures the essential properties needed to
    create TorchRL specs and validate tensor shapes throughout the system.
    """
    bounds: tuple[float, float] | None = Field(
        default     = None,
        description = "Optional (min, max) bounds for bounded tensors"
    )
    dtype: Literal["float32", "int64", "bool"] = Field(
        default     = "float32",
        description = "Tensor data type"
    )
    shape: list[str] = Field(
        description = "Shape dimensions using 'N' for agent count and 'd' for spatial dims"
    )


class FlockModel(BaseModel, extra="forbid"):
    """
    Unified configuration for the thermal drone flock.
    
    Combines agent physical properties, collective behavior parameters,
    and spatial settings into a single coherent configuration used across
    simulation, control, and safety components.
    
    The thermal properties govern agent survivability and safety constraints.
    The maximum temperature T_max defines the Control Barrier Function's safety
    boundary:
    
        h(𝐱) = T_max - T(𝐱)
    
    The thermal time constant τ models heat dissipation dynamics using an RC
    thermal circuit analogy, allowing estimation of core temperature from surface
    measurements:
    
        T_core ≈ T_skin - τ · dT_skin/dt
    
    The communication range R_comm determines the dynamic neighborhood graph
    G_t = (V, E_t) at each timestep t, where edges exist between agents i and j
    when:
    
        ||𝐱_i - 𝐱_j|| ≤ R_comm
    
    This metric-based connectivity contrasts with topological neighborhoods used
    in biological flocks (typically 6-7 nearest neighbors regardless of distance).
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        gt          = 1,
        description = "Number of agents N in the flock."
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        gt          = 0,
        description = (
            "Metric distance R_comm in meters defining the neighborhood graph "
            "connectivity threshold."
        )
    )
    formation_scale_factor: PositiveFloat = Field(
        default     = 0.5,
        gt          = 0,
        le          = 1,
        description = (
            "Scaling factor γ ∈ (0, 1] applied to initial formations as a "
            "fraction of communication range, controlling flock density."
        )
    )
    initial_formation: Literal["cube", "sphere", "random"] = Field(
        default     = "sphere",
        description = (
            "Geometric arrangement of agents at simulation initialization."
        )
    )
    max_temperature: PositiveFloat = Field(
        default     = 475.0,
        description = (
            "Maximum survivable agent temperature T_max in Kelvin (K), "
            "defining the hard safety boundary h(𝐱) for the CBF."
        )
    )
    spatial_dims: Literal[2, 3] = Field(
        default     = 3,
        description = (
            "Number of spatial dimensions d for the simulation workspace ℝ^d."
        )
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "RC thermal model time constant τ in seconds, characterizing the "
            "agent's heat dissipation rate."
        )
    )


class ObservationSpace(BaseModel, extra="forbid"):
    """
    Defines the observation structure for the thermal flock system.
    
    This model specifies the complete state representation available to agents,
    including kinematics, thermal data, and communication topology. The structure
    is used throughout the system for data consistency and can be converted to
    TorchRL specs for environment interfaces.
    
    Shape notation:
    - N : Number of agents (resolved from FlockModel.agent_count)
    - d : Spatial dimensions (resolved from FlockModel.spatial_dims)
    """
    battery: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "1"], bounds=(0.0, 1.0)),
        description = "Energy remaining per agent"
    )
    edge_index: TensorSpec = Field(
        default     = TensorSpec(shape=["2", "N*(N-1)"], dtype="int64"),
        description = "Dynamic graph connectivity for inter-agent communication"
    )
    position: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "d"]),
        description = "Spatial coordinates in ℝᵈ"
    )
    temperature: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "1"], bounds=(0.0, float("inf"))),
        description = "Thermal state per agent in Kelvin"
    )
    temperature_grad: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "d"]),
        description = "Thermal gradient ∇T at agent positions"
    )
    velocity: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "d"]),
        description = "Motion vectors in ℝᵈ"
    )
    wind: TensorSpec = Field(
        default     = TensorSpec(shape=["N", "d"]),
        description = "Environmental wind field at agent positions"
    )