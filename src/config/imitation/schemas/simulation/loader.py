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
        description = (
            "Enable stochastic environmental variations including wind perturbations "
            "and temperature noise to improve policy generalization."
        )
    )
    fire_heat_variable: str = Field(
        default     = "GRNHFX",
        description = (
            "NetCDF variable identifier for ground-level heat flux from wildfire "
            "combustion, measured in W/m² in WRF-Fire outputs."
        )
    )
    temperature_noise_std: NonNegativeFloat = Field(
        default     = 0.5,
        description = (
            "Temperature noise standard deviation σ_T in Kelvin for domain "
            "randomization, simulating measurement uncertainty and turbulence."
        )
    )
    u_wind_variable: str = Field(
        default     = "U",
        description = (
            "NetCDF variable identifier for eastward wind component U on staggered "
            "Arakawa-C grid, requiring interpolation to cell centers."
        )
    )
    v_wind_variable: str = Field(
        default     = "V",
        description = (
            "NetCDF variable identifier for northward wind component V on staggered "
            "Arakawa-C grid, requiring interpolation to cell centers."
        )
    )
    w_wind_variable: str = Field(
        default     = "W", 
        description = (
            "NetCDF variable identifier for vertical wind component W, critical "
            "for modeling thermal updrafts and fire-induced convection."
        )
    )
    wind_noise_std: NonNegativeFloat = Field(
        default     = 0.1,
        description = (
            "Wind noise standard deviation σ_w in m/s for stochastic perturbations, "
            "modeling atmospheric turbulence and measurement uncertainty."
        )
    )