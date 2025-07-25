"""
Loader configuration schema.

This module defines the configuration for WRF data loading, including
NetCDF file parameters and processing options.
"""
from pydantic import BaseModel, Field, FilePath, NonNegativeFloat
from typing   import Optional


class LoaderModel(BaseModel, extra="forbid"):
    """
    Configuration for WRF data loader.
    
    This model defines data structure parameters (variable names, processing 
    options) and caching configuration for the Moisseeva (2020) dataset.
    
    The dataset contains 147 NetCDF files totaling 5.33 TB. The simulation
    uses staggered grids where U, V, W wind components are offset from cell centers.
    """
    data_path: Optional[FilePath] = Field(
        default     = None,
        description = (
            "Path to WRF-Fire NetCDF dataset file. Defaults to first available: "
            "wrf-sfire/*.nc, then data/samples/wrf_sample.nc"
        )
    )
    domain_randomization: bool = Field(
        default     = True,
        description = "Enable domain randomization for robustness."
    )
    fire_heat_variable: str = Field(
        default     = "GRNHFX",
        description = "NetCDF variable name for ground heat flux from fire."
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