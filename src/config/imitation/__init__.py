"""
Imitation learning configuration system.

This module provides the entry point for Hydra-based configuration, orchestrating
all domain-specific configurations for the training pipeline. The configuration
system uses hydra-zen's hierarchical pattern for type-safe, modular configuration.

The configuration is organized by domains:
- controller    : Expert policy and safety systems
- lightning     : Training infrastructure and optimization
- monitoring    : Metrics and event tracking
- simulation    : Environment and physics
- visualization : 3D visualization with PyVista

All user-facing configuration is exposed through Pydantic models, while
pre-built components are hidden in the _system namespace.

Experiment tracking is handled by Weights & Biases (configured in lightning.wandb).
"""
from .controller.builds    import CONTROLLER_USER_CONFIG,    CONTROLLER_SYSTEM_BUILDS
from .lightning.builds     import LIGHTNING_USER_CONFIG,     LIGHTNING_SYSTEM_BUILDS
from .monitoring.builds    import MONITORING_USER_CONFIG,    MONITORING_SYSTEM_BUILDS
from .simulation.builds    import SIMULATION_USER_CONFIG,    SIMULATION_SYSTEM_BUILDS
from .visualization.builds import VISUALIZATION_USER_CONFIG, VISUALIZATION_SYSTEM_BUILDS
from hydra_zen             import make_config

ImitationConfig = make_config(
    controller    = CONTROLLER_USER_CONFIG,
    lightning     = LIGHTNING_USER_CONFIG,
    monitoring    = MONITORING_USER_CONFIG,
    simulation    = SIMULATION_USER_CONFIG,
    visualization = VISUALIZATION_USER_CONFIG,
    _system       = {
        **CONTROLLER_SYSTEM_BUILDS,
        **LIGHTNING_SYSTEM_BUILDS,
        **MONITORING_SYSTEM_BUILDS,
        **SIMULATION_SYSTEM_BUILDS,
        **VISUALIZATION_SYSTEM_BUILDS
    },
    hydra_defaults = ["_self_", {"override hydra/hydra_logging": "none"}],
    hydra = {
        "run": {
            "dir": (
                "outputs/"
                "${oc.select:lightning.wandb.run_name,"
                "'${now:%Y-%m-%d}/${now:%H-%M-%S}'}"
            )
        }
    }
)

__all__ = ["ImitationConfig"]
