"""
System configuration and validation schemas for the Thermur CLI.

This module defines models for system requirements, validation messages,
and integration status tracking.
"""
from pydantic import BaseModel, Field


class SystemModel(BaseModel, extra="forbid"):
    """
    Defines system requirements and paths for the CLI.
    
    This model specifies the Python version requirements, package dependencies,
    and important file paths used by the CLI. It ensures compatibility and
    helps with system validation during startup.
    """
    config_search_paths: list[str] = Field(
        default     = ["configs", "src/configs"],
        description = "Paths to search for Hydra configuration files"
    )
    disk_thresholds: tuple[int, int] = Field(
        default     = (5, 20),
        description = "Low and medium disk space thresholds in GB"
    )
    memory_thresholds: tuple[int, int] = Field(
        default     = (4, 8),
        description = "Low and medium memory thresholds in GB"
    )
    min_python_version: tuple[int, int] = Field(
        default     = (3, 9),
        description = "Minimum required Python version as (major, minor)"
    )
    project_root_markers: list[str] = Field(
        default     = ["pyproject.toml", "src/thermur"],
        description = "Files/directories that indicate the project root"
    )
    required_packages: list[str] = Field(
        default     = ["torch", "mujoco", "hydra-core", "wandb"],
        description = "List of required Python packages for full functionality"
    )


class ValidationModel(BaseModel, extra="forbid"):
    """
    System validation requirements and diagnostic messages.
    
    This model holds validation criteria and messages for thermal flock
    simulation and training environments.
    """
    all_validations_passed: str = Field(
        default     = "✅ All validations passed!",
        description = "Message when all validations succeed"
    )
    config_fail_msg: str = Field(
        default     = "Configuration validation failed:",
        description = "Header for config validation failures"
    )
    config_issues_found: str = Field(
        default     = "Configuration issues found:",
        description = "Header for config issues list"
    )
    config_validation_passed: str = Field(
        default     = "Configuration validation passed!",
        description = "Message for successful config validation"
    )
    force_override_tip: str = Field(
        default     = "Use --force to override or fix the issues above.",
        description = "Tip about using force flag"
    )
    review_issues_tip: str = Field(
        default     = "Review the issues above before training",
        description = "Tip to review validation issues"
    )
    system_ready: str = Field(
        default     = "Your system is ready for training",
        description = "Message when system is ready"
    )
    validation_with_warnings: str = Field(
        default     = "⚠️  Validation completed with warnings",
        description = "Message for validation with warnings"
    )


class WandbDisplayModel(BaseModel, extra="forbid"):
    """
    Weights & Biases integration display configuration.
    
    This model centralizes environment variable keys and display strings
    for wandb integration in the CLI.
    """
    api_key_env: str = Field(
        default     = "WANDB_API_KEY",
        description = "Environment variable for wandb API key"
    )
    default_project: str = Field(
        default     = "thermur",
        description = "Default project name for wandb"
    )
    details_api_key: str = Field(
        default     = "[white]Ready to track[/white]",
        description = "Details text when API key is set"
    )
    details_connected: str = Field(
        default     = "[cyan]@{user}[/cyan]",
        description = "Details template when connected"
    )
    details_not_connected: str = Field(
        default     = "[yellow]Run 'wandb login'[/yellow]",
        description = "Details text when not connected"
    )
    details_not_installed: str = Field(
        default     = "[yellow]pip install wandb[/yellow]",
        description = "Details text when not installed"
    )
    entity_env: str = Field(
        default     = "WANDB_ENTITY",
        description = "Environment variable for wandb entity"
    )
    example_projects: list[str] = Field(
        default = [
            "thermal-flock-v1",
            "drone-flocking-experiments",
            "heat-aware-navigation",
            "imitation-learning-tests",
        ],
        description = "Example project names for wandb"
    )
    mode_env: str = Field(
        default     = "WANDB_MODE",
        description = "Environment variable for wandb mode"
    )
    status_api_key: str = Field(
        default     = "[green]✅ API Key Set[/green]",
        description = "Status text when API key is set"
    )
    status_connected: str = Field(
        default     = "[green]✅ Connected[/green]",
        description = "Status text when connected"
    )
    status_not_connected: str = Field(
        default     = "[yellow]⚠️  Not Connected[/yellow]",
        description = "Status text when not connected"
    )
    status_not_installed: str = Field(
        default     = "[red]❌ Not Installed[/red]",
        description = "Status text when not installed"
    )