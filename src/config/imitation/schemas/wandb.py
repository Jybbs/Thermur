"""
Weights & Biases configuration for experiment tracking.

This module defines the configuration for W&B integration during training,
enabling experiment tracking, metric logging, and model checkpointing.
"""
from pydantic import BaseModel, Field
from typing   import Literal


class WandbModel(BaseModel, extra="forbid"):
    """
    Configuration for Weights & Biases experiment tracking.
    
    W&B provides comprehensive experiment tracking for machine learning workflows,
    including metric logging, hyperparameter tracking, model versioning, and
    visualization. This configuration controls how Lightning integrates with W&B
    during training and how the CLI accesses W&B for monitoring.
    
    The mode parameter allows flexible deployment:
    - "online": Full cloud synchronization for collaborative work
    - "offline": Local logging for air-gapped environments  
    - "disabled": No W&B integration (useful for debugging)
    
    The API key is read from an environment variable to avoid hardcoding
    credentials in configuration files.
    """
    api_key: str = Field(
        default     = "WANDB_API_KEY",
        description = (
            "Environment variable name containing the W&B API key for "
            "authentication - keeps credentials out of config files."
        )
    )
    log_model: Literal["all", "false"] | bool = Field(
        default     = "all",
        description = (
            "Model checkpoint logging policy - 'all' saves every checkpoint, "
            "'false' or False disables model logging to save bandwidth."
        )
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = (
            "W&B tracking mode - online syncs to cloud, offline stores locally, "
            "disabled skips W&B integration entirely."
        )
    )
    project: str = Field(
        default     = "thermur-imitation",
        description = (
            "W&B project name for organizing experiments - groups related training "
            "runs for easier comparison and analysis."
        )
    )
