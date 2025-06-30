"""
Physics and simulation model.

This module defines the unified configuration for the physical simulation
environment, including MuJoCo settings and thermal field interpolation.
"""
from __future__ import annotations
from pathlib    import Path
from pydantic   import BaseModel, Field, PositiveFloat


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
    simulation_step: PositiveFloat = Field(
        default     = 0.05,
        description = "Physics simulation timestep Δt in seconds."
    )
    assets_dir: Path = Field(
        default     = Path("src/thermur/simulation/assets"),
        description = "Directory containing MuJoCo XML model files."
    )
    bounds_min: list[float] = Field(
        default     = [0.0, 0.0, 0.0],
        description = "Minimum coordinates [x_min, y_min, z_min] of workspace."
    )
    bounds_max: list[float] = Field(
        default     = [50.0, 50.0, 20.0],
        description = "Maximum coordinates [x_max, y_max, z_max] of workspace."
    )
    data_source: Path = Field(
        default     = Path("data/environment/sample_field.nc"),
        description = "Path to NetCDF file containing thermal field data."
    )
    epsilon: PositiveFloat = Field(
        default     = 1e-6,
        description = "Finite difference step size ε for gradient computation."
    )
    fallback_temperature: float = Field(
        default     = 20.0,
        description = "Default temperature T_default when interpolation fails."
    )
    temperature_variable: str = Field(
        default     = "temperature",
        description = "NetCDF variable name containing temperature data."
    )
    x_dimension: str = Field(
        default     = "x",
        description = "NetCDF dimension name for x-coordinate."
    )
    y_dimension: str = Field(
        default     = "y",
        description = "NetCDF dimension name for y-coordinate."
    )
    z_dimension: str = Field(
        default     = "z",
        description = "NetCDF dimension name for z-coordinate."
    )