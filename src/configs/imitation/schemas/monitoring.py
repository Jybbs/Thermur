"""
Monitoring and logging models.

This module defines the Pydantic models for application logging
and experiment tracking parameters.
"""
from pathlib  import Path
from pydantic import BaseModel, Field
from typing   import Literal, Optional


class LoggingModel(BaseModel, extra="forbid"):
    """
    Configuration for the Loguru logging setup.

    This class controls the verbosity, format, and destinations of log
    messages generated throughout the application.
    """
    colorize: bool = Field(
        default     = True,
        description = "Whether to use colorized log output for the console."
    )
    diagnose: bool = Field(
        default     = False,
        description = "Whether to add exception tracebacks to the log for debugging."
    )
    enqueue: bool = Field(
        default     = True,
        description = "Whether to make file logging non-blocking and thread-safe."
    )
    file_path: Optional[Path] = Field(
        default     = Path("logs/thermur.log"),
        description = "Path to the log file. If None, file logging is disabled."
    )
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default     = "INFO",
        description = "The minimum log level to be processed and displayed."
    )
    retention: str = Field(
        default     = "7 days",
        description = "Log file retention policy (e.g., '10 days', '1 month')."
    )
    rotation: str = Field(
        default     = "10 MB",
        description = "Log file rotation policy (e.g., '500 MB', '12:00')."
    )


class WandbModel(BaseModel, extra="forbid"):
    """
    Configuration for Weights & Biases experiment tracking.

    These parameters control how training runs are logged and organized for
    visualization, analysis, and comparison.
    """
    entity: Optional[str] = Field(
        default     = None,
        description = "The W&B entity (username or team name)."
    )
    mode: Literal["online", "offline", "disabled"] = Field(
        default     = "online",
        description = "The W&B run mode."
    )
    project: str = Field(
        default     = "thermur",
        description = "The W&B project name to log runs into."
    )