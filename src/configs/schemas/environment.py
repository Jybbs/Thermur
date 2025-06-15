"""
Environment and physics models.

This module defines the Pydantic models for the simulation environment
and expert controller parameters.
"""
from pydantic import BaseModel, Field


class EnvironmentModel(BaseModel, extra="forbid"):
    """
    Configuration for the simulation environment.

    This class specifies the environment to be instantiated, including its
    dynamics and the source of the physical data (wind and temperature fields)
    that it will provide to the agents.
    """
    data_source: str = Field(
        default     = "data/wrfout_d01.nc",
        description = (
            "Path to the environmental data source (e.g., NetCDF from "
            "WRF-Fire)."
        )
    )
    name: str = Field(
        default     = "WRF-Fire-Env-v0",
        description = "The registered name of the Gymnasium environment to use."
    )
    simulation_step: float = Field(
        default     = 0.05,
        gt          = 0,
        description = (
            "The duration of a single simulation physics step (Δt) in seconds."
        )
    )


class ExpertPolicyModel(BaseModel, extra="forbid"):
    """
    Defines the weights for the handcrafted 'expert' flocking controller.

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
