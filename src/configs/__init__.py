"""
Configuration management for the Thermur project.
"""
from hydra.core.config_store import ConfigStore
from .entry_points.imitation_train import imitation_train_config, register_imitation_training_config


__all__ = ["register_imitation_training_config"]
