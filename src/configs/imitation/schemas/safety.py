"""
Safety system configuration.

This module defines the unified configuration for Control Barrier Functions
and quadratic program solvers used to ensure thermal safety constraints.
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
    # Control Barrier Function parameters
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
    
    # Debug and monitoring
    log_violations: bool = Field(
        default     = True,
        description = "Log safety constraint violations to monitor performance."
    )
    
    # QP solver settings
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