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

base_config_store.add_to_hydra_store()

# Register main config
main_store = store()
main_store(ImitationConfig, group="config", name="imitation", package="_global_")
main_store.add_to_hydra_store()

__all__ = ["ImitationConfig"]