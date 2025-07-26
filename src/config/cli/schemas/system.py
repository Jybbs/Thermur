"""
System inspection and validation configuration schemas.

This module defines configuration for system requirements, validation rules,
and environment inspection used by the SystemInspector helper.
"""
from pydantic import BaseModel, Field


class SystemModel(BaseModel, extra="forbid"):
    """
    System inspection and validation configuration.
    
    This model defines system requirements and validation rules used
    by the SystemInspector to check environment compatibility.
    """
    cuda_preferred: bool = Field(
        default     = True,
        description = "Whether CUDA GPU acceleration is preferred for training."
    )
    dataset_validation: dict[str, float] = Field(
        default = {
            "min_size_gb"    : 0.1,
            "max_size_gb"    : 10000.0,
            "warning_size_gb": 100.0,
        },
        description = "Dataset size validation thresholds in gigabytes."
    )
    mujoco_min_version: str = Field(
        default     = "2.3.0",
        description = "Minimum MuJoCo version required for physics simulation."
    )
    python_min_version: tuple[int, int] = Field(
        default     = (3, 9),
        description = "Minimum Python version required as (major, minor) tuple."
    )
    required_packages: list[str] = Field(
        default = [
            "torch",
            "pytorch_lightning",
            "torchrl",
            "mujoco",
            "hydra-core",
            "wandb",
            "rich",
            "typer",
        ],
        description = "List of required Python packages for system validation."
    )