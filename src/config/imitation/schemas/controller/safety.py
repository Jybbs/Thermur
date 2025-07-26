"""
Safety filter configuration model.

This module defines the configuration for the safety filtering system that
ensures all control commands respect thermal constraints using Control
Barrier Functions (CBF).
"""
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt
from typing   import Literal


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
            "Control deviation threshold in m/s for detecting CBF interventions, used "
            "to track when safety filter modifies nominal commands by ||u* - u_nom|| > δ."
        )
    )
    cbf_alpha: PositiveFloat = Field(
        default     = 2.5,
        description = (
            "Class-K function parameter α > 0 defining exponential safety convergence "
            "rate via CBF constraint ∇h(x)ᵀu ≥ -αh(x) where h(x) = T_max - T(x)."
        )
    )
    log_violations: bool = Field(
        default     = True,
        description = (
            "Enable logging of thermal safety violations and CBF activations for "
            "debugging controller behavior and monitoring safety-critical events during training."
        )
    )
    qp_eps: PositiveFloat = Field(
        default     = 1e-6,
        description = (
            "Convergence tolerance ε for the quadratic program solver determining when "
            "||u^(k+1) - u^(k)|| < ε indicates optimal solution found."
        )
    )
    qp_max_iter: PositiveInt = Field(
        default     = 100,
        description = (
            "Maximum solver iterations before termination, balancing solution quality "
            "against real-time computational constraints in the control loop."
        )
    )
    qp_on_failure: Literal["zero", "nominal", "raise"] = Field(
        default     = "zero",
        description = (
            "Fallback strategy when QP fails: 'zero' applies zero control for safety, "
            "'nominal' uses unfiltered input, 'raise' propagates exception for debugging."
        )
    )
    qp_verbose: bool = Field(
        default     = False,
        description = (
            "Enable detailed quadratic program solver logging including iteration counts, "
            "constraint violations, and convergence metrics for algorithm debugging."
        )
    )