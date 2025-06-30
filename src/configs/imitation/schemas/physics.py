"""
Physics and simulation configuration.

This module defines configurations for the physical simulation environment,
including MuJoCo settings, thermal dynamics, and world properties.
"""
from __future__ import annotations

from pathlib  import Path
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Literal


class PhysicsConfig(BaseModel, extra="forbid"):
    """
    Physical simulation and environment configuration.
    
    Controls the MuJoCo physics engine settings, simulation timestep,
    and environmental properties used throughout the system.
    """
    # Simulation settings
    simulation_step: PositiveFloat = Field(
        default     = 0.05,
        description = "Physics simulation timestep Δt in seconds."
    )
    episode_length: PositiveInt = Field(
        default     = 500,
        description = "Maximum steps per episode before reset."
    )
    
    # Environment paths
    assets_dir: Path = Field(
        default     = Path("assets/mujoco"),
        description = "Directory containing MuJoCo XML model files."
    )
    
    # World properties
    world_bounds: tuple[float, float, float] = Field(
        default     = (100.0, 100.0, 50.0),
        description = "World boundary dimensions (x, y, z) in meters."
    )
    gravity: tuple[float, float, float] = Field(
        default     = (0.0, 0.0, -9.81),
        description = "Gravity vector (x, y, z) in m/s²."
    )
    
    # Thermal environment
    ambient_temperature: float = Field(
        default     = 20.0,
        description = "Ambient temperature T_amb in Celsius."
    )
    thermal_noise_scale: float = Field(
        default     = 0.1,
        description = "Scale factor for thermal measurement noise."
    )


class ThermalFieldConfig(BaseModel, extra="forbid"):
    """
    Thermal field interpolation and data source configuration.
    
    Defines how environmental temperature data is loaded, interpolated,
    and made available to agents for navigation decisions.
    """
    # Data source settings
    data_path: Path = Field(
        default     = Path("data/thermal_field.npz"),
        description = "Path to thermal field data file."
    )
    
    # Interpolation parameters
    epsilon: PositiveFloat = Field(
        default     = 1e-8,
        description = "Numerical stability constant for gradient computation."
    )
    fallback_temperature: float = Field(
        default     = 300.0,
        description = "Temperature value used outside data bounds (Kelvin)."
    )
    temperature_variable: str = Field(
        default     = "temperature",
        description = "Variable name in data file containing temperature field."
    )
    
    # Field dimensions
    x_dimension: str = Field(
        default     = "x",
        description = "Variable name for x-coordinate grid."
    )
    y_dimension: str = Field(
        default     = "y", 
        description = "Variable name for y-coordinate grid."
    )
    z_dimension: str = Field(
        default     = "z",
        description = "Variable name for z-coordinate grid."
    )