"""
Flock configuration model.

This module defines the unified configuration for the multi-agent flock,
including physical properties, collective behavior, and spatial settings.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


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