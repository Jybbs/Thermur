"""
Hydra-zen builder for Weights & Biases configuration.

This module provides the configuration builder for W&B experiment tracking,
creating a validated configuration for Lightning's WandbLogger.
"""
from config.imitation.schemas.wandb import WandbModel
from hydra_zen                      import builds


build_wandb = builds(
    WandbModel,
    populate_full_signature = True,
    zen_dataclass           = {
        "module"   : "src.config.imitation.factories.wandb",
        "cls_name" : "WandbBuild"
    }
)
"""
Builder for W&B experiment tracking configuration.

Creates a validated configuration that controls how Lightning integrates
with Weights & Biases for metric logging, hyperparameter tracking, and
experiment organization during training.
"""