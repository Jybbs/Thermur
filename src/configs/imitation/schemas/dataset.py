"""
Dataset configuration schema.

This module defines the configuration for WRF-Fire datasets, including both
download parameters and data structure definitions for the NetCDF files.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveInt


class DatasetModel(BaseModel, extra="forbid"):
    """
    Configuration for WRF-Fire dataset management.
    
    This model combines dataset acquisition parameters (download, caching)
    with data structure definitions (variable names, processing options).
    It provides a single configuration point for all WRF-Fire data needs.
    
    The Moisseeva (2020) dataset contains 147 NetCDF files totaling 5.33 TB,
    making subset downloads essential for development. WRF uses staggered
    grids where U, V, W wind components are offset from cell centers.
    """
    cache_dir: Path = Field(
        default     = Path("data/wrf_cache"),
        description = "Local directory for caching downloaded NetCDF files."
    )
    dataset_name: str = Field(
        default     = "moisseeva_2020",
        description = "Name of the dataset configuration to use."
    )
    domain_randomization: bool = Field(
        default     = True,
        description = "Enable domain randomization for robustness."
    )
    endpoint_id: str = Field(
        default     = "dd39c220-356d-4f06-a8e5-77016c648ca4",
        description = "Globus endpoint UUID for the dataset."
    )
    fire_heat_variable: str = Field(
        default     = "GRNHFX",
        description = "NetCDF variable name for ground heat flux from fire."
    )
    max_files: PositiveInt = Field(
        default     = 2,
        description = "Maximum number of files to download for training."
    )
    temperature_noise_std: NonNegativeFloat = Field(
        default     = 0.5,
        description = "Standard deviation of Gaussian noise added to temperature."
    )
    u_wind_variable: str = Field(
        default     = "U",
        description = "NetCDF variable name for x-wind component."
    )
    v_wind_variable: str = Field(
        default     = "V",
        description = "NetCDF variable name for y-wind component."
    )
    w_wind_variable: str = Field(
        default     = "W", 
        description = "NetCDF variable name for z-wind component."
    )
    wind_noise_std: NonNegativeFloat = Field(
        default     = 0.1,
        description = "Standard deviation of noise added to wind vectors."
    )