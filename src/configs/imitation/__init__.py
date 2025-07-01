"""
Imitation learning configuration domain.

This package contains all configuration-related code for imitation learning,
including schemas, factories, and workloads organized by functionality.
"""
from .workloads.imitation import register_configs

__all__ = ["register_configs"]