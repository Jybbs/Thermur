"""
Simulation domain schemas for Pydantic validation.

This module consolidates all simulation configuration models including
physics settings, data loading, and environment parameters.
"""
from pydantic import BaseModel, Field
from pydantic import NonNegativeFloat, PositiveFloat


class LoaderModel(BaseModel, extra="forbid"):
    """
    Configuration for WRF data loader.

    This model defines data structure parameters (variable names, processing
    options) and caching configuration for the Moisseeva (2020) dataset.

    The dataset contains 147 NetCDF files totaling 5.33 TB. The simulation
    uses staggered grids where U, V, W wind components are offset from cell centers.
    """
    data_path: str = Field(
        default     = "auto",
        description = (
            "Path to WRF-Fire NetCDF dataset file. Use 'auto' to automatically find "
            "wrf-sfire/*.nc, then fallback to data/samples/wrf_sample.nc"
        )
    )
    domain_randomization: bool = Field(
        default     = True,
        description = (
            "Enable stochastic environmental variations including wind perturbations "
            "and temperature noise to improve policy generalization."
        )
    )
    temperature_noise_std: NonNegativeFloat = Field(
        default     = 0.5,
        description = (
            "Temperature noise standard deviation σ_T in Kelvin for domain "
            "randomization, simulating measurement uncertainty and turbulence."
        )
    )
    wind_noise_std: NonNegativeFloat = Field(
        default     = 0.1,
        description = (
            "Wind noise standard deviation σ_w in m/s for stochastic perturbations, "
            "modeling atmospheric turbulence and measurement uncertainty."
        )
    )


class PhysicsModel(BaseModel, extra="forbid"):
    """
    Unified physics and environmental simulation configuration.

    Controls the physics simulation settings, timestep, spatial bounds,
    and thermal field interpolation parameters used throughout the system.

    The simulation operates on a discrete timestep Δt, advancing the
    physics state using Euler integration. The thermal field
    provides spatially-varying temperature data T(𝐱) through
    interpolation of gridded measurements.

    Gradient computation uses finite differences with step size ε:

        ∇T(𝐱) ≈ [T(𝐱 + εê_i) - T(𝐱 - εê_i)] / 2ε

    where ê_i are the standard basis vectors in ℝ^d.
    """
    bounds_max: list[float] = Field(
        default     = [50.0, 50.0, 20.0],
        description = (
            "Upper bounds [x_max, y_max, z_max] in meters defining the rectangular "
            "workspace volume where agents can safely operate."
        )
    )
    bounds_min: list[float] = Field(
        default     = [0.0, 0.0, 0.0],
        description = (
            "Lower bounds [x_min, y_min, z_min] in meters defining the rectangular "
            "workspace origin, typically [0, 0, 0] for simplicity."
        )
    )
    epsilon: PositiveFloat = Field(
        default     = 1e-6,
        description = (
            "Numerical step size ε for finite difference gradient approximation, "
            "balancing accuracy against floating-point precision limits."
        )
    )
    fallback_temperature: float = Field(
        default     = 20.0,
        description = (
            "Fallback temperature T_default in Celsius used when interpolation "
            "fails outside data bounds or during initialization."
        )
    )
    drag_coefficient: NonNegativeFloat = Field(
        default     = 0.1,
        description = (
            "Quadratic drag coefficient Cd for aerodynamic resistance modeling, "
            "where F_drag = -Cd * v * |v| simulates air resistance effects."
        )
    )
    gravity: PositiveFloat = Field(
        default     = 9.81,
        description = (
            "Gravitational acceleration g in m/s². Used for physics calculations "
            "and energy consumption estimation."
        )
    )
    initial_spacing_factor: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Multiplicative factor for initial Fibonacci lattice spacing relative to "
            "communication range, ensuring strong initial k-NN connectivity."
        )
    )
    max_speed: PositiveFloat = Field(
        default     = 20.0,
        description = (
            "Maximum agent velocity v_max in m/s enforced as a hard constraint, "
            "representing physical limitations of drone propulsion systems."
        )
    )
    simulation_step: PositiveFloat = Field(
        default     = 0.05,
        description = (
            "Integration timestep Δt in seconds for Euler physics integration, "
            "balancing accuracy with real-time computational constraints."
        )
    )
