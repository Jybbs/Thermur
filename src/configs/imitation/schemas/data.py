"""
Data acquisition and management schemas.

This module defines configuration models for acquiring and managing
wildfire simulation datasets, particularly large-scale WRF-Fire outputs
like the Moisseeva (2020) LES plume dataset.
"""
from pathlib  import Path
from pydantic import BaseModel, Field, PositiveInt


class DataAcquisitionModel(BaseModel, extra="forbid"):
    """
    Configuration for wildfire dataset acquisition and management.
    
    Controls how the system downloads and manages subsets of large
    wildfire simulation datasets. The Moisseeva (2020) dataset contains
    147 NetCDF files totaling 5.33 TB, so this configuration enables
    downloading manageable subsets for development and training.
    
    The acquisition system uses Globus for efficient large-scale data
    transfers and maintains a local manifest to track downloaded files.
    """
    cache_dir: Path = Field(
        default     = Path("data/wrf_cache"),
        description = "Local directory for caching downloaded NetCDF files."
    )
    dataset_name: str = Field(
        default     = "moisseeva_2020",
        description = "Name of the dataset configuration to use."
    )
    endpoint_id: str = Field(
        default     = "dd39c220-356d-4f06-a8e5-77016c648ca4",
        description = "Globus endpoint UUID for the dataset."
    )
    max_files: PositiveInt = Field(
        default     = 2,
        description = "Maximum number of files to download for training."
    )
    max_size_gb: float = Field(
        default     = 50.0,
        gt          = 0,
        description = "Maximum total download size in gigabytes."
    )


class WRFDataModel(BaseModel, extra="forbid"):
    """
    Configuration for WRF-Fire NetCDF data structure and variables.
    
    Defines the expected structure and variable names for WRF-Fire
    output files. WRF uses staggered grids where U, V, W components
    are offset from cell centers, requiring special handling.
    
    Standard WRF variables:
    - T: Perturbation potential temperature (actual = 300 + T)
    - U, V, W: Wind components on staggered grid points
    - GRNHFX: Ground heat flux from fire
    - QVAPOR: Water vapor mixing ratio
    """
    domain_randomization: bool = Field(
        default     = True,
        description = "Enable domain randomization for robustness."
    )
    temperature_noise_std: float = Field(
        default     = 0.5,
        ge          = 0,
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
    wind_noise_std: float = Field(
        default     = 0.1,
        ge          = 0,
        description = "Standard deviation of noise added to wind vectors."
    )