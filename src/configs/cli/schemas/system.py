"""
System configuration and validation schemas for the Thermur CLI.

This module defines models for system requirements, validation messages,
and integration status tracking.
"""
from pydantic import BaseModel, Field


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
