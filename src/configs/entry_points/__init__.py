"""
Entry point configurations for the Thermur project.

This package contains focused configurations for specific entry points,
assembling only the necessary components for each use case.
"""
from .imitation_train import imitation_train_config, register_imitation_training_config

__all__ = [
    "imitation_train_config", 
    "register_imitation_training_config"
]
