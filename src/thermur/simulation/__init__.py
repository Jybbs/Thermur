"""
Simulation components for the Thermur project.

This package provides the fundamental building blocks including the environment,
data structures, and geometric utilities.
"""
from __future__   import annotations
from .environment import SimulationEnv
from .geometry    import compute_edge_index

__all__ = [
    "compute_edge_index",
    "SimulationEnv",
]
