"""
Simulation domain schemas for Pydantic validation.

This module consolidates all simulation configuration models including
physics settings, data loading, and environment parameters.
"""
from pathlib  import Path
from pydantic import BaseModel, DirectoryPath, Field, FilePath
from pydantic import NonNegativeFloat, PositiveFloat
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
        default     = Path("src/thermur/simulation/assets"),
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