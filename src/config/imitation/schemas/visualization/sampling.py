"""
Visualization sampling configuration.

This module provides models for configuring how continuous simulation data
is discretized and sampled for visualization purposes.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt


class GridModel(BaseModel, extra="forbid"):
    """
    Configures sampling grid parameters for visualization data.
    
    These settings control how continuous simulation data is discretized
    for visualization purposes, affecting both visual quality and performance.
    """
    padding: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Buffer distance in meters added to flock bounding box for grid "
            "generation, preventing edge artifacts in volume rendering."
        )
    )
    temperature_resolution: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default     = (20, 20, 20),
        description = (
            "Voxel grid dimensions (nx, ny, nz) for temperature field interpolation, "
            "balancing visual smoothness against memory usage and rendering speed."
        )
    )
    wind_resolution: PositiveInt = Field(
        default     = 5,
        description = (
            "Grid points per dimension for wind vector visualization, creating a "
            "regular 3D lattice of arrow glyphs showing airflow patterns."
        )
    )