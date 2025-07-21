"""
Hydra-zen builders for monitoring and logging configurations.

This module provides factory functions that create Hydra-compatible
configurations for logging and experiment tracking components.
"""
from ..schemas import LoggingModel
from ...cli.schemas.application import WandbModel
from hydra_zen import builds

build_logging = builds(
    LoggingModel,
    populate_full_signature = True,
    zen_dataclass = {
        "module"   : "src.configs.imitation.factories.monitoring",
        "cls_name" : "LoggingBuild"
    }
)
"""
Builder for Loguru logging configuration.

Creates a configuration for application-wide logging with sensible defaults
for both console and file output.
"""
