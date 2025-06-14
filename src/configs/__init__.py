"""
Configuration module for the Thermur application.

This module provides Pydantic models for validation and hydra-zen builders
for component instantiation.
"""
from .pydantic import (
    AgentConfig,
    CBFConfig,
    CheckpointConfig,
    CollectorConfig,
    EnvironmentConfig,
    ExpertPolicyConfig,
    GNNConfig,
    LoggingConfig,
    PolicyConfig,
    QPSolverConfig,
    ReplayBufferConfig,
    SafetyConfig,
    SwarmConfig,
    TrainConfig,
    WandbConfig,
)

# Lazy import for AppConfig to avoid circular dependencies
def __getattr__(name):
    if name == "AppConfig":
        from .app import get_app_config
        return get_app_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Pydantic models
    "AgentConfig",
    "CBFConfig",
    "CheckpointConfig",
    "CollectorConfig",
    "EnvironmentConfig",
    "ExpertPolicyConfig",
    "GNNConfig",
    "LoggingConfig",
    "PolicyConfig",
    "QPSolverConfig",
    "ReplayBufferConfig",
    "SafetyConfig",
    "SwarmConfig",
    "TrainConfig",
    "WandbConfig",
    # Main config
    "AppConfig",
]
