"""
Safety mechanism models.

This module defines the Pydantic models for Control Barrier Function
and Quadratic Program solver parameters.
"""
from pydantic import BaseModel, Field
from typing   import Literal


class CBFModel(BaseModel, extra="forbid"):
    """
    Parameters for the Control Barrier Function (CBF) safety filter.

    The CBF guarantees forward invariance of the safety set C, ensuring agents
    do not exceed `max_temperature` via the constraint: ḣ(𝐱) ≥ -αh(𝐱). This
    is solved in real-time via a Quadratic Program (QP).
    """
    alpha: float = Field(
        default     = 0.5,
        gt          = 0,
        description = (
            "Class-K function gain (α) for the CBF safety constraint, "
            "controlling convergence to the safe set."
        )
    )


class QPSolverModel(BaseModel, extra="forbid"):
    """
    Parameters for the qpth Quadratic Program (QP) solver.

    These settings control the behavior and numerical precision of the
    differentiable QP solver used in the safety filter.
    """
    eps: float = Field(
        default     = 1e-7,
        gt          = 0,
        description = "Tolerance for constraint satisfaction in the QP solver."
    )
    max_iter: int = Field(
        default     = 20,
        gt          = 0,
        description = "Maximum number of iterations for the QP solver."
    )
    on_failure: Literal["error", "use_nominal"] = Field(
        default     = "error",
        description = (
            "Action to take if the QP solver fails. 'error' raises an "
            "exception, 'use_nominal' falls back to the original unsafe action."
        )
    )
