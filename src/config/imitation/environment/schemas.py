"""
Simulation domain schemas for Pydantic validation.

This module consolidates all simulation configuration models including
physics settings, data loading, and environment parameters.
"""
from pydantic import BaseModel, Field
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt


class DatasetModel(BaseModel, extra="forbid"):
    """
    Configuration for dataset generation and data sources.

    This model defines parameters for generating offline demonstration datasets
    including episode structure, data sources, and generation scope.
    """
    frames_per_episode: PositiveInt = Field(
        default     = 1000,
        description = (
            "Number of frames per demonstration episode. Longer episodes "
            "capture extended temporal dependencies in flocking behavior."
        )
    )
    sample_url: str = Field(
        default     = (
            "https://huggingface.co/datasets/Jybbs/sfire-samples/"
            "resolve/main/samples.tar.gz"
        ),
        description = (
            "URL for downloading sample WRF-SFIRE dataset when no local data exists. "
            "Points to a curated 1.5GB sample containing moderate intensity scenarios."
        )
    )
    total_frames: PositiveInt = Field(
        default     = 200_000,
        description = (
            "Total demonstration frames to generate across all scenarios. "
            "Determines the size of the offline dataset for expert trajectory collection."
        )
    )


class PhysicsModel(BaseModel, extra="forbid"):
    """
    Unified physics and environmental simulation configuration.

    Controls the physics simulation settings, timeframe, spatial bounds,
    and thermal field interpolation parameters used throughout the system.

    The simulation operates on a discrete timeframe Δt, advancing the
    physics state using Euler integration. The thermal field
    provides spatially-varying temperature data T(𝐱) through
    interpolation of gridded measurements.

    Gradient computation uses finite differences with step size ε:

        ∇T(𝐱) ≈ [T(𝐱 + εê_i) - T(𝐱 - εê_i)] / 2ε

    where ê_i are the standard basis vectors in ℝ^d.
    """
    bounds_max: list[float] = Field(
        default     = [500.0, 500.0, 40.0],
        description = (
            "Upper bounds [x_max, y_max, z_max] in meters defining the rectangular "
            "workspace volume where agents can safely operate."
        )
    )
    bounds_min: list[float] = Field(
        default     = [-500.0, -500.0, 0.0],
        description = (
            "Lower bounds [x_min, y_min, z_min] in meters defining the rectangular "
            "workspace origin, typically [0, 0, 0] for simplicity."
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
    initial_altitude: PositiveFloat = Field(
        default     = 20.0,
        description = (
            "Initial altitude in meters for flock center of mass. Agents start "
            "at this height above ground level to simulate aerial murmuration."
        )
    )
    initial_spacing_factor: PositiveFloat = Field(
        default     = 0.1,
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
    timeframe: PositiveFloat = Field(
        default     = 0.05,
        description = (
            "Integration timeframe Δt in seconds for Euler physics integration, "
            "balancing accuracy with real-time computational constraints."
        )
    )
    wind_coupling_coefficient: NonNegativeFloat = Field(
        default     = 0.05,
        description = (
            "Wind coupling coefficient Cw for aerodynamic force from relative wind, "
            "where F_wind = Cw * (v_wind - v_agent). Reduced from drag coefficient "
            "to model partial sheltering effects within the flock formation."
        )
    )
