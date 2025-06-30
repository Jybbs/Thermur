"""
Control system configuration.

This module defines configurations for the flocking controller,
including Reynolds rules weights and numerical stability parameters.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat


class ControlConfig(BaseModel, extra="forbid"):
    """
    Unified control system configuration.
    
    Combines Reynolds flocking weights and numerical parameters into
    a single configuration for the expert controller that generates
    demonstrations for imitation learning.
    """
    # Reynolds rule weights
    weight_cohesion: NonNegativeFloat = Field(
        default     = 0.3,
        description = "Weight ω_coh for cohesion force (flock centering)."
    )
    weight_separation: NonNegativeFloat = Field(
        default     = 0.8,
        description = "Weight ω_sep for separation force (collision avoidance)."
    )
    weight_alignment: NonNegativeFloat = Field(
        default     = 0.5,
        description = "Weight ω_align for velocity alignment (coordinated motion)."
    )
    weight_thermal: NonNegativeFloat = Field(
        default     = 2.0,
        description = "Weight ω_thermal for thermal gradient avoidance."
    )
    
    # Numerical stability parameters
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum distance ε_dist in meters to prevent division by zero "
            "in separation calculations."
        )
    )
    max_velocity: PositiveFloat = Field(
        default     = 10.0,
        description = "Maximum velocity v_max in m/s for control saturation."
    )
    velocity_damping: PositiveFloat = Field(
        default     = 0.9,
        le          = 1.0,
        description = (
            "Velocity damping factor β ∈ (0, 1] applied to control commands "
            "for smoother trajectories."
        )
    )