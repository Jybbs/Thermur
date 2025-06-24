"""
Pydantic models for 3D visualization configuration.

This module provides models for configuring the real-time 3D visualization
capabilities of the Thermur simulation. These include rendering options,
display properties, and toggles for various visualization elements like
agent representations, thermal fields, safety boundaries, and graph topology.
"""
from pydantic import BaseModel, Field


class VisualizationModel(BaseModel, extra="forbid"):
    """
    Configures the 3D visualization settings for the simulation.
    
    Controls rendering of swarm state, thermal conditions, wind fields, and other
    simulation elements for debugging, assessment, and analysis purposes. Most
    options can be toggled at runtime.
    """
    colormap: str = Field(
        default     = "plasma",
        description = "Colormap for thermal visualization (e.g., 'plasma', 'inferno')."
    )
    dark_mode: bool = Field(
        default     = True,
        description = "Whether to use a dark theme with black background."
    )
    enabled: bool = Field(
        default     = False,
        description = "Master switch to enable/disable visualization during execution."
    )
    glyph_size: float = Field(
        default     = 0.15,
        description = "Size scaling factor for agent glyphs."
    )
    glyph_type: str = Field(
        default     = "sphere",
        description = "Type of glyph used to represent agents ('sphere' or 'arrow')."
    )
    grid_padding: float = Field(
        default     = 2.0,
        description = "Extra space around the swarm's bounding box for visualization grids."
    )
    show_agents: bool = Field(
        default     = True,
        description = "Whether to render the swarm agents as 3D glyphs."
    )
    show_graph: bool = Field(
        default     = False,
        description = "Whether to visualize the swarm's communication graph connectivity."
    )
    show_safety: bool = Field(
        default     = False,
        description = "Whether to visualize the thermal safety boundary (T_max isosurface)."
    )
    show_thermal: bool = Field(
        default     = True,
        description = "Whether to color agents based on their sensed temperature."
    )
    show_trails: bool = Field(
        default     = False,
        description = "Whether to show motion trails behind agents."
    )
    show_wind: bool = Field(
        default     = False,
        description = "Whether to visualize the wind field with vector glyphs."
    )
    temp_grid_resolution: tuple = Field(
        default     = (20, 20, 20),
        description = "Resolution of the temperature field sampling grid (nx, ny, nz)."
    )
    wind_grid_resolution: int = Field(
        default     = 5,
        description = "Resolution of the wind field sampling grid in each dimension."
    )
    window_size: tuple = Field(
        default     = (1024, 768),
        description = "Size of the visualization window in pixels (width, height)."
    )
    window_title: str = Field(
        default     = "Thermur Simulation",
        description = "Title displayed in the visualization window frame."
    )
