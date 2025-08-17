"""
Imitation learning configuration system.

This module provides the entry point for Hydra-based configuration, orchestrating
all domain-specific configurations for the training pipeline. The configuration
system uses hydra-zen's hierarchical pattern for type-safe, modular configuration.

The configuration is organized by domains:
- controller : Expert policy and safety systems
- simulation : Environment and physics
- training   : Training infrastructure, optimization, and metrics

All user-facing configuration is exposed through Pydantic models, while
pre-built components are hidden in the _system namespace.

Experiment tracking is handled by Weights & Biases (configured in training.wandb).
"""
from .controller.builds import CONTROLLER_USER_CONFIG, CONTROLLER_SYSTEM_BUILDS
from .simulation.builds import SIMULATION_USER_CONFIG, SIMULATION_SYSTEM_BUILDS
from .training.builds   import TRAINING_USER_CONFIG,   TRAINING_SYSTEM_BUILDS
from datetime           import datetime
from hydra_zen          import make_config
from omegaconf          import OmegaConf
from typing             import Callable

resolver: Callable[[str | None], str] = lambda run_name: (
    f"outputs/{run_name}" if run_name 
    else (
        f"outputs/{datetime.now().strftime('%Y-%m-%d')}/"
        f"{datetime.now().strftime('%H-%M-%S')}"
    )
)

OmegaConf.register_new_resolver(
    "output_dir",
    resolver,
    replace = True
)

ImitationConfig = make_config(
    controller = CONTROLLER_USER_CONFIG,
    simulation = SIMULATION_USER_CONFIG,
    training   = TRAINING_USER_CONFIG,
    _system    = {
        **CONTROLLER_SYSTEM_BUILDS,
        **SIMULATION_SYSTEM_BUILDS,
        **TRAINING_SYSTEM_BUILDS
    },
    hydra = {
        "run": {
            "dir": "${output_dir:${training.wandb.run_name}}"
        }
    }
)

__all__ = ["ImitationConfig"]
