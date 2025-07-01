"""
Control system model.

This module defines the unified configuration for the flocking controller,
including Reynolds rules weights and numerical stability parameters.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat


class ControlModel(BaseModel, extra="forbid"):
    """
    Unified control system configuration for expert flocking behavior.
    
    Combines Reynolds flocking weights and numerical parameters into
    a single configuration for the expert controller that generates
    demonstrations for imitation learning.
    
    The controller computes nominal actions 𝐮_nom from the negative
    gradient of a potential function U, where:
    
        𝐮_nom = -∇_x U(S_t)
    
    The potential U combines classical Reynolds rules with thermal constraints:
    
        U = ω_coh · U_coh + ω_sep · U_sep + ω_align · U_align + ω_thermal · U_thermal
    
    where the individual potentials are:
        - Cohesion:   U_coh   = (1/2) · Σ_j∈N(i) ||𝐱_i - 𝐱_j||²
        - Separation: U_sep   = Σ_j∈N(i) 1/||𝐱_i - 𝐱_j||
        - Alignment:  U_align = (1/2) · Σ_j∈N(i) ||𝐯_i - 𝐯_j||²
        - Thermal:    U_thermal = 1/(T_max - T_i)
    """
    epsilon: PositiveFloat = Field(
        default     = 1e-8,
        description = (
            "Numerical stability constant ε for safe division operations."
        )
    )
    gradient_step: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Finite difference step size δ for estimating temperature "
            "gradients when analytical gradients are unavailable."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum distance ε_dist in meters to prevent division by zero "
            "in separation force calculations."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Scaling factor λ_thermal for thermal force magnitude, allowing "
            "fine-tuning relative to Reynolds forces."
        )
    )
    w_alignment: NonNegativeFloat = Field(
        default     = 0.8,
        description = (
            "Weight ω_align for velocity alignment. Higher values encourage "
            "agents to match velocities with their neighbors."
        )
    )
    w_cohesion: NonNegativeFloat = Field(
        default     = 1.0,
        description = (
            "Weight ω_coh for cohesion force. Higher values encourage agents "
            "to move toward the neighborhood center of mass."
        )
    )
    w_separation: NonNegativeFloat = Field(
        default     = 1.5,
        description = (
            "Weight ω_sep for separation force. Higher values create stronger "
            "repulsion between nearby agents to avoid collisions."
        )
    )
    w_thermal: NonNegativeFloat = Field(
        default     = 2.0,
        description = (
            "Weight ω_thermal for thermal avoidance. Higher values create "
            "stronger repulsion from high-temperature regions."
        )
    )