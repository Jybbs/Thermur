"""
Safety mechanism models.

This module defines the Pydantic models for Control Barrier Function
and Quadratic Program solver parameters.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


class CBFModel(BaseModel, extra="forbid"):
    """
    Parameters for the Control Barrier Function (CBF) safety filter.

    The CBF guarantees forward invariance of the safety set C, ensuring agents
    do not exceed `max_temperature` via the constraint: ḣ(𝐱) ≥ -αh(𝐱). This
    is solved in real-time via a Quadratic Program (QP).
    """
    activation_tolerance: PositiveFloat = Field(
        default     = 1e-5,
        description = (
            "Numerical tolerance for detecting CBF activations. If the norm of "
            "the difference between safe and nominal actions exceeds this value, "
            "the CBF is considered active for that agent."
        )
    )
    alpha: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Class-K function gain (α) for the CBF safety constraint, "
            "controlling convergence to the safe set."
        )
    )
    debug_mode: bool = Field(
        default     = False,
        description = (
            "Enable visualization of the T_max isotherm and detailed logging "
            "of CBF activations during simulation."
        )
    )


class QPSolverModel(BaseModel, extra="forbid"):
    """
    Parameters for the qpth Quadratic Program (QP) solver.

    These settings control the behavior and numerical precision of the
    differentiable QP solver used in the safety filter.
    """
    eps: PositiveFloat = Field(
        default     = 1e-7,
        description = "Tolerance for constraint satisfaction in the QP solver."
    )
    max_iter: PositiveInt = Field(
        default     = 20,
        description = "Maximum number of iterations for the QP solver."
    )
    on_failure: Literal["error", "use_nominal"] = Field(
        default     = "error",
        description = (
            "Action to take if the QP solver fails. 'error' raises an "
            "exception, 'use_nominal' falls back to the original unsafe action."
        )
    )
