"""
Imitation learning configuration system.

This module provides the entry point for Hydra-based configuration, orchestrating
all domain-specific configurations for the training pipeline. The configuration
system uses hydra-zen's hierarchical pattern for type-safe, modular configuration.

The configuration is organized by domains:
- controller  : Expert policy and safety systems
- environment : WRF data and trajectory physics
- training    : Training infrastructure, optimization, and metrics

All user-facing configuration is exposed through Pydantic models, while
pre-built components are hidden in the _system namespace.

Experiment tracking is handled by Weights & Biases (configured in training.wandb).
"""
from .controller.builds  import CONTROLLER_USER_CONFIG,  CONTROLLER_SYSTEM_BUILDS
from .environment.builds import ENVIRONMENT_USER_CONFIG, ENVIRONMENT_SYSTEM_BUILDS
from .training.builds    import TRAINING_USER_CONFIG,    TRAINING_SYSTEM_BUILDS
from hydra_zen           import make_config

ImitationConfig = make_config(
    controller  = CONTROLLER_USER_CONFIG,
    environment = ENVIRONMENT_USER_CONFIG,
    training    = TRAINING_USER_CONFIG,
    _system     = {
        **CONTROLLER_SYSTEM_BUILDS,
        **ENVIRONMENT_SYSTEM_BUILDS,
        **TRAINING_SYSTEM_BUILDS
    },
    hydra      = {
        "hydra_logging" : {"level": "DISABLED"},
        "job_logging"   : {"level": "DISABLED"},
        "output_subdir" : None,
        "run"           : {"dir": "."},
    }
)

__all__ = ["ImitationConfig"]
