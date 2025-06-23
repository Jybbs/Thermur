"""
Environment and physics models.

This module defines the Pydantic models for the simulation environment.
"""
from pathlib  import Path
from pydantic import BaseModel, Field


class EnvironmentModel(BaseModel, extra="forbid"):
    """
    Configuration for the simulation environment.

    This class specifies the environment to be instantiated, including its
    dynamics and the source of the physical data (wind and temperature fields)
    that it will provide to the agents.
    """
    assets_dir: Path = Field(
        default     = Path("src/thermur/simulation/assets"),
        description = "Directory containing simulation asset files like MuJoCo XML models."
    )
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


class ThermalInterpolationModel(BaseModel, extra="forbid"):
    """
    Parameters for thermal data interpolation and gradient calculation.
    
    These parameters control how the continuous temperature field is sampled
    and how temperature gradients are calculated for arbitrary agent positions.
    The implementation uses vectorized operations to efficiently process
    batches of position queries.
    """
    epsilon: float = Field(
        default     = 0.1,
        gt          = 0,
        description = "Distance (ε) used for finite difference gradient calculation in meters."
    )
    fallback_temperature: float = Field(
        default     = 300.0,
        description = "Default temperature value when interpolation fails or produces NaN."
    )
    fill_value: float = Field(
        default     = float('nan'),
        description = "Value to use for out-of-bounds positions in interpolation."
    )
    temperature_variable: str = Field(
        default     = "T",
        description = "Name of the temperature variable in the dataset."
    )
    x_dimension: str = Field(
        default     = "x",
        description = "Name of the x-dimension coordinate in the dataset."
    )
    y_dimension: str = Field(
        default     = "y",
        description = "Name of the y-dimension coordinate in the dataset."
    )
    z_dimension: str = Field(
        default     = "z",
        description = "Name of the z-dimension coordinate in the dataset."
    )
