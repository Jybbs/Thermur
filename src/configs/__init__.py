"""
Configuration management for the Thermur project.

This package provides Pydantic models and Hydra-zen builders for
imitation learning training configuration.
"""
from .entry_points import register_imitation_training_config

__all__ = ["register_imitation_training_config"]
