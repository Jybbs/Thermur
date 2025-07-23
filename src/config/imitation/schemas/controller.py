"""
Control system model.

This module defines the unified configuration for the flocking controller,
including Reynolds rules weights and numerical stability parameters.
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing   import Literal


class ControllerModel(BaseModel, extra="forbid"):
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

class SafetyModel(BaseModel, extra="forbid"):
    """
    Unified safety system configuration.
    
    Combines Control Barrier Function parameters and QP solver settings
    into a single configuration for the safety filtering system that
    ensures all control commands respect thermal constraints.
    """
    activation_tolerance: NonNegativeFloat = Field(
        default     = 3.0,
        description = (
            "Temperature buffer δ in Kelvin before CBF activation. "
            "CBF triggers when T > T_max - δ."
        )
    )
    cbf_alpha: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "CBF class-K function parameter α controlling the rate of "
            "exponential safety convergence: ḣ(x) + α·h(x) ≥ 0."
        )
    )
    log_violations: bool = Field(
        default     = True,
        description = "Log safety constraint violations to monitor performance."
    )
    qp_eps: PositiveFloat = Field(
        default     = 1e-6,
        description = "Numerical tolerance ε for QP solver convergence."
    )
    qp_max_iter: PositiveInt = Field(
        default     = 100,
        description = "Maximum iterations for iterative QP solver."
    )
    qp_on_failure: Literal["zero", "nominal", "raise"] = Field(
        default     = "zero",
        description = (
            "Action on QP solver failure: 'zero' (safe stop), "
            "'nominal' (pass through), or 'raise' (exception)."
        )
    )
    qp_verbose: bool = Field(
        default     = False,
        description = "Enable verbose QP solver output for debugging."
    )