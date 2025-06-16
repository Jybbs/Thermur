"""
Flocking behavior models.

This module defines the Pydantic models for the Reynolds-based flocking
parameters and numerical stability values used in the controller.
"""
from pydantic import BaseModel, Field


class ReynoldsWeightsModel(BaseModel, extra="forbid"):
    """
    Defines the weights for the handcrafted flocking controller.

    This controller's nominal action, 𝐮_nom, is derived from the negative
    gradient of a synthetic potential energy function, U = -∇ₓU(Sₜ). These
    parameters weight the components of that function, which are based on
    classic Reynolds rules and our thermal constraints.
    - Cohesion   : U_coh   ∝ Σ||xᵢ - xⱼ||²
    - Separation : U_sep   ∝ Σ 1/||xᵢ - xⱼ||
    - Alignment  : U_align ∝ Σ||vᵢ - vⱼ||²
    - Thermal    : U_therm ∝ 1/(T_{max} - Tᵢ)
    """
    w_alignment: float = Field(
        default     = 0.8,
        description = (
            "Weight for the alignment potential. Higher values encourage agents "
            "to match velocity with neighbors."
        )
    )
    w_cohesion: float = Field(
        default     = 1.0,
        description = (
            "Weight for the cohesion potential. Higher values encourage agents "
            "to form a tighter group."
        )
    )
    w_separation: float = Field(
        default     = 1.5,
        description = (
            "Weight for the separation potential. Higher values create more "
            "space between nearby agents."
        )
    )
    w_thermal: float = Field(
        default     = 2.0,
        description = (
            "Weight for the thermal potential. Higher values create a stronger "
            "repulsion from high-temperature regions."
        )
    )


class FlockingModel(BaseModel, extra="forbid"):
    """
    Numerical parameters for the flocking controller's computations.
    
    These parameters ensure stable and accurate calculation of the Reynolds
    rule forces by preventing numerical issues like division by zero and
    ensuring proper scaling of gradient estimates.
    """
    epsilon: float = Field(
        default     = 1e-8,
        gt          = 0,
        description = (
            "Small constant to prevent division by zero in distance and "
            "temperature margin calculations."
        )
    )
    gradient_step: float = Field(
        default     = 0.1,
        gt          = 0,
        description = (
            "Step size for finite difference calculations when estimating "
            "temperature gradients."
        )
    )
    temperature_scaling: float = Field(
        default     = 1.0,
        gt          = 0,
        description = (
            "Scaling factor for thermal repulsion force magnitude, allowing "
            "adjustment relative to other forces."
        )
    )
