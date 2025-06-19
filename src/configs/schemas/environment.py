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
