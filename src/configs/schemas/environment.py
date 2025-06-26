"""
Environment models.

This module defines the Pydantic models for the simulation environment,
including physical parameters, data sources, and interpolation strategies.
"""
from pydantic import BaseModel, DirectoryPath, Field, FilePath, PositiveFloat
from typing   import Literal


class EnvironmentModel(BaseModel, extra="forbid"):
    """
    Configuration for the simulation environment parameters.
    
    These settings control the physical simulation properties, including
    the spatial bounds, time stepping, and data sources for wind and
    temperature fields.
    """
    assets_dir: DirectoryPath = Field(
        default     = "src/thermur/simulation/assets",
        description = "Directory containing MJCF assets for the MuJoCo simulation."
    )
    bounds_max: list[float] = Field(
        default     = [50.0, 50.0, 20.0],
        description = "Maximum coordinates [x, y, z] of the simulation bounds."
    )
    bounds_min: list[float] = Field(
        default     = [0.0, 0.0, 0.0],
        description = "Minimum coordinates [x, y, z] of the simulation bounds."
    )
    data_source: FilePath = Field(
        default     = "data/environment/sample_field.nc",
        description = "Path to the NetCDF file containing wind and temperature data."
    )
    simulation_step: PositiveFloat = Field(
        default     = 0.05,
        description = "Time step (in seconds) for the physics simulation."
    )
    

class ThermalInterpolationModel(BaseModel, extra="forbid"):
    """
    Configuration for thermal field interpolation.
    
    These parameters control how continuous thermal values are computed from
    discrete grid data, affecting the accuracy and smoothness of temperature
    queries throughout the simulation space.
    """
    bounds_padding: PositiveFloat = Field(
        default     = 0.1,
        description = "Relative padding to add to bounds for interpolation stability."
    )
    fill_value: float = Field(
        default     = float("nan"),
        description = "Value to return for queries outside the data bounds."
    )
    method: Literal["linear", "nearest", "cubic"] = Field(
        default     = "linear",
        description = "Interpolation method for thermal field queries."
    )
