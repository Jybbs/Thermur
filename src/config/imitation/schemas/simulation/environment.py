"""
Physics and simulation environment configuration.

This module defines the unified configuration for the physical simulation
environment, including MuJoCo settings and thermal field interpolation.
"""
from pydantic import BaseModel, DirectoryPath, Field, PositiveFloat


class PhysicsModel(BaseModel, extra="forbid"):
    """
    Unified physics and environmental simulation configuration.
    
    Controls the MuJoCo physics engine settings, simulation timestep,
    spatial bounds, and thermal field interpolation parameters used
    throughout the system.
    
    The simulation operates on a discrete timestep Δt, advancing the
    physics state according to the equations of motion. The thermal
    field provides spatially-varying temperature data T(𝐱) through
    interpolation of gridded measurements.
    
    Gradient computation uses finite differences with step size ε:
    
        ∇T(𝐱) ≈ [T(𝐱 + εê_i) - T(𝐱 - εê_i)] / 2ε
    
    where ê_i are the standard basis vectors in ℝ^d.
    """
    assets_dir: DirectoryPath = Field(
        default     = "src/thermur/simulation/assets",
        description = (
            "Root directory containing MuJoCo XML model definitions for drone "
            "dynamics, including mesh files and physical parameters."
        )
    )
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
    gravity: PositiveFloat = Field(
        default     = 9.81,
        description = (
            "Gravitational acceleration g in m/s². Used for physics calculations "
            "and energy consumption estimation."
        )
    )
    simulation_step: PositiveFloat = Field(
        default     = 0.05,
        description = (
            "Integration timestep Δt in seconds for MuJoCo physics solver, "
            "balancing accuracy with real-time computational constraints."
        )
    )
    temperature_variable: str = Field(
        default     = "temperature",
        description = (
            "NetCDF variable identifier for temperature field data, matching "
            "the naming convention in WRF-Fire output files."
        )
    )
    x_dimension: str = Field(
        default     = "x",
        description = (
            "NetCDF dimension identifier for east-west spatial coordinate, "
            "typically 'x' or 'west_east' depending on data source."
        )
    )
    y_dimension: str = Field(
        default     = "y",
        description = (
            "NetCDF dimension identifier for north-south spatial coordinate, "
            "typically 'y' or 'south_north' depending on data source."
        )
    )
    z_dimension: str = Field(
        default     = "z",
        description = (
            "NetCDF dimension identifier for vertical spatial coordinate, "
            "typically 'z' or 'bottom_top' for atmospheric data."
        )
    )