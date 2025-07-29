"""
Imitation learning configuration system.

This module provides the entry point for Hydra-based configuration, orchestrating
all domain-specific configurations for the training pipeline. The configuration
system uses hydra-zen's store pattern for type-safe, modular configuration.

The configuration is organized by domains:
- controller    : Expert policy and safety systems
- lightning     : Training infrastructure and optimization
- monitoring    : Metrics and event tracking  
- simulation    : Environment and physics
- visualization : 3D visualization with PyVista

Experiment tracking is handled by Weights & Biases (configured in lightning.wandb).
"""
from .controller.stores    import controller
from .lightning.stores     import lightning
from .monitoring.stores    import monitoring
from .simulation.stores    import simulation
from .visualization.stores import visualization
from hydra_zen             import make_config

# Create the top-level config using the `thermur_make_all` calls from each domain
ImitationConfig = make_config(
    hydra_defaults=[
        "_self_",
        {"controller"    : "all"},
        {"lightning"     : "all"},
        {"monitoring"    : "all"},
        {"simulation"    : "all"},
        {"visualization" : "all"},
    ]
)

__all__ = ["ImitationConfig"]
