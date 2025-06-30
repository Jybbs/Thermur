"""
Imitation learning workload configurations.

This module contains the top-level workload configurations that compose
all necessary components for imitation learning training.
"""
from .imitation import imitation_config, register_configs

__all__ = [
    "imitation_config",
    "register_configs",
]