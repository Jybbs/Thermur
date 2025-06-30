"""
Visualization system for the Thermur simulation environment.

This package provides 3D visualization capabilities for the swarm simulation,
enabling real-time visualization of agent motion, thermal conditions, wind fields,
safety constraints, and communication topologies. It serves as a critical tool
for debugging, qualitative assessment, and insight generation.
"""
from __future__  import annotations
from .renderers  import *
from .sampling   import *
from .visualizer import Visualizer

__all__ = [
    # Main visualization interface
    "Visualizer",

    # Rendering functions
    "render_agents",
    "render_communication_graph",
    "render_safety_boundary",
    "render_temperature_field",
    "render_wind_field",

    # Sampling utilities
    "compute_grid_bounds",
    "create_coordinate_axes",
    "create_temperature_grid",
    "create_wind_grid",
]
