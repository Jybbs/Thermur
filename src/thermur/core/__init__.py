"""
Core components for the Thermur project.

This package provides the fundamental building blocks including the environment,
data structures, and geometric utilities.
"""
from .cli        import cli_main
from .env        import ThermurEnv
from .geometry   import compute_edge_index
from .structures import SwarmData, SwarmDataSpec

__all__ = [
    "cli_main",
    "compute_edge_index",
    "SwarmData",
    "SwarmDataSpec",
    "ThermurEnv",
]
