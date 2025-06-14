"""
Core components for the Thermur project.

This package provides the fundamental building blocks including the environment,
data structures, and geometric utilities.
"""
from .cli        import app
from .simulation import SimulationEnv
from .geometry   import compute_edge_index
from .structures import SwarmData, SwarmDataSpec

__all__ = [
    "app",
    "compute_edge_index",
    "SwarmData",
    "SwarmDataSpec",
    "SimulationEnv",
]
