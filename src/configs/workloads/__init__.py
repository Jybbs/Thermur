"""
Workload configurations for Thermur.

This package contains configurations for different workloads like training and evaluation.
"""
from .imitation import imitation_config, register_configs

__all__ = ["imitation_config", "register_configs"]
