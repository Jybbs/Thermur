"""
Simulation domain schemas for Pydantic validation.

This module consolidates all simulation configuration models including
physics settings, data loading, and environment parameters.
"""
from pydantic import BaseModel, Field
from pydantic import NonNegativeFloat, PositiveFloat, PositiveInt


class DatasetModel(BaseModel, extra="forbid"):
    """
    Configuration for offline expert dataset generation.

    Defines parameters for generating expert trajectories from WRF-SFIRE data.
    Each trajectory maintains consistent environmental conditions by using a
    single WRF snapshot for its entire duration, preventing mid-trajectory
    discontinuities while ensuring dataset diversity across snapshots.
    """
    sample_url: str = Field(
        default     = (
            "https://huggingface.co/datasets/Jybbs/sfire-samples/"
            "resolve/main/samples.tar.gz"
        ),
        description = (
            "URL for downloading sample WRF-SFIRE dataset when no local "
            "data exists. Points to a curated 1.5GB sample containing "
            "moderate intensity fire scenarios."
        )
    )
    trajectories_per_snapshot: PositiveInt = Field(
        default     = 100,
        description = (
            "Number of expert trajectories to generate per WRF snapshot. "
            "Each trajectory uses different initial conditions (agent "
            "positions and velocities) but the same environmental snapshot, "
            "ensuring physically consistent conditions throughout."
        )
    )
    trajectory_duration: PositiveFloat = Field(
        default     = 60.0,
        description = (
            "Duration of each continuous trajectory in seconds. Longer "
            "trajectories capture extended temporal dynamics of collective "
            "flocking behavior under consistent environmental conditions."
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
    air_density: PositiveFloat = Field(
        default     = 1.225,
        description = (
            "Air density ρ in kg/m³ at standard sea level conditions "
            "(15°C, 101.325 kPa). Standard atmospheric value from "
            "ISO 2533:1975 Standard Atmosphere."
        )
    )
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
    cross_sectional_area: PositiveFloat = Field(
        default     = 0.002,
        description = (
            "Frontal cross-sectional area A in m² for aerodynamic drag calculation. "
            "Typical value for small songbird body cross-section viewed head-on. "
            "Approximately π × (0.025)² for a ~5cm body diameter."
        )
    )
    drag_coefficient: NonNegativeFloat = Field(
        default     = 0.075,
        description = (
            "Dimensionless drag coefficient C_d for aerodynamic force calculation. "
            "Mid-range value for birds in cruising flight per aerodynamic literature."
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
