"""
Agent and swarm models.

This module defines the Pydantic models for agent physical properties
and swarm collective behavior parameters.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


class AgentModel(BaseModel, extra="forbid"):
    """
    Defines the physical and sensory properties of a single agent.

    This class consolidates parameters that govern an agent's thermal
    survivability and its function as a perceptible information display. The
    thermal properties are critical for modeling the agent's internal state
    and enforcing safety guarantees.

    The `max_temperature` provides T_{max} for the Control Barrier Function's
    safety boundary: h(𝐱) = T_{max} - T(𝐱). The `thermal_time_constant` (τ) is
    used in the agent's internal RC thermal model to estimate core temperature
    from skin temperature: T_core ≈ T_skin - τ ⋅ dT_skin/dt.
    """
    led_color_space: Literal["CIELAB", "Oklab"] = Field(
        default     = "CIELAB",
        description = (
            "The perceptually-uniform color space for mapping temperature to a "
            "visible color."
        )
    )
    max_temperature: PositiveFloat = Field(
        default     = 500.0,
        description = (
            "Maximum survivable agent temperature in Fahrenheit (°F), defining "
            "the hard safety boundary h(𝐱) for the CBF."
        )
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = (
            "RC thermal model time constant (τ) in seconds, used to estimate "
            "internal temperature."
        )
    )


class SwarmModel(BaseModel, extra="forbid"):
    """
    Configures the collective properties and initial state of the agent swarm.

    These parameters define the scale of the multi-agent system and the rules
    for local interaction. The `communication_range` is particularly critical
    as it defines the dynamic graph topology Gₜ = (V, Eₜ) at each timestep.
    This metric-based neighborhood is a practical starting point, while natural
    flocks often use a fixed topological neighborhood (e.g., 6-7 nearest agents).
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        gt          = 1,
        description = "The number of agents (N) in the swarm."
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        description = (
            "The metric distance in meters for defining the topological "
            "neighborhood graph."
        )
    )
    formation_scale_factor: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Scaling factor applied to initial agent formations, as a fraction "
            "of the communication range. Controls the density of the swarm."
        )
    )
    initial_formation: Literal["sphere", "cube"] = Field(
        default     = "sphere",
        description = (
            "The geometric formation of the swarm at the start of the simulation."
        )
    )
    spatial_dims: PositiveInt = Field(
        default     = 3,
        ge          = 2,
        description = (
            "The number of spatial dimensions in the simulation (e.g., 2 for 2D, 3 "
            "for 3D)."
        )
    )
