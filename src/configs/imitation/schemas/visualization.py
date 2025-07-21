"""
Pydantic models for 3D visualization configuration.

This module provides models for configuring the real-time 3D visualization
capabilities of the Thermur simulation. These include rendering options,
display properties, and toggles for various visualization elements like
agent representations, thermal fields, safety boundaries, and graph topology.
"""
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Annotated, Literal

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
RGBColor  = tuple[UnitFloat, UnitFloat, UnitFloat]


class ColorModel(BaseModel, extra="forbid"):
    """
    Configures color settings for visualization elements.
    
    These settings control the color mapping and default colors used
    throughout the visualization system.
    """
    agent_default: RGBColor = Field(
        default     = (0.2, 0.2, 0.8),
        description = "Default RGB color for agents when not using thermal coloring."
    )
    colormap: Literal["plasma", "inferno", "viridis", "magma", "cividis", "turbo", "thermal"] = Field(
        default     = "plasma",
        description = "Colormap for thermal visualization."
    )
    graph_default: RGBColor = Field(
        default     = (0.7, 0.7, 0.9),
        description = "Default RGB color for communication graph edges."
    )
    safety_default: RGBColor = Field(
        default     = (0.9, 0.3, 0.3),
        description = "Default RGB color for safety boundary."
    )
    scalar_bar_position_x: UnitFloat = Field(
        default     = 0.88,
        description = "Horizontal position of the scalar bar (0=left, 1=right)."
    )
    scalar_bar_position_y: UnitFloat = Field(
        default     = 0.25,
        description = "Vertical position of the scalar bar (0=bottom, 1=top)."
    )
    scalar_bar_title: str = Field(
        default     = "Temperature",
        description = "Title displayed on the temperature scalar bar."
    )
    trail_default: RGBColor = Field(
        default     = (0.8, 0.8, 0.8),
        description = "Default RGB color for agent motion trails."
    )
    wind_default: RGBColor = Field(
        default     = (0.7, 0.7, 0.7),
        description = "Default RGB color for wind field arrows."
    )


class GlyphModel(BaseModel, extra="forbid"):
    """
    Configures glyph rendering parameters for agent visualization.
    
    Glyphs are the 3D objects used to represent agents in the visualization.
    These settings control their appearance and behavior.
    """
    arrow_scale: PositiveFloat = Field(
        default     = 0.1,
        description = "Scaling factor for arrow glyphs based on velocity magnitude."
    )
    size: PositiveFloat = Field(
        default     = 0.15,
        description = "Size scaling factor for agent glyphs."
    )
    trail_length: NonNegativeInt = Field(
        default     = 5,
        description = "Number of points to include in motion trails."
    )
    type: Literal["sphere", "arrow"] = Field(
        default     = "sphere",
        description = "Type of glyph to represent agents."
    )


class GridModel(BaseModel, extra="forbid"):
    """
    Configures sampling grid parameters for visualization data.
    
    These settings control how continuous simulation data is discretized
    for visualization purposes, affecting both visual quality and performance.
    """
    padding: PositiveFloat = Field(
        default     = 2.0,
        description = "Extra space around the flock's bounding box for grids."
    )
    temperature_resolution: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default     = (20, 20, 20),
        description = "Resolution for temperature field sampling grid (nx, ny, nz)."
    )
    wind_resolution: PositiveInt = Field(
        default     = 5,
        description = "Resolution for wind field sampling grid in each dimension."
    )


class OpacityModel(BaseModel, extra="forbid"):
    """
    Configures opacity values for different visualization elements.
    
    Opacity values range from 0.0 (fully transparent) to 1.0 (fully opaque).
    These settings allow fine-tuning the visual layering of simulation elements.
    """
    agents: UnitFloat = Field(
        default     = 1.0,
        description = "Opacity of agent glyphs."
    )
    graph: UnitFloat = Field(
        default     = 0.5,
        description = "Opacity of communication graph edges."
    )
    safety: UnitFloat = Field(
        default     = 0.3,
        description = "Opacity of safety boundary isosurface."
    )
    trails: UnitFloat = Field(
        default     = 0.5,
        description = "Opacity of agent motion trails."
    )
    wind: UnitFloat = Field(
        default     = 0.8,
        description = "Opacity of wind field arrows."
    )


class VisualizationModel(BaseModel, extra="forbid"):
    """
    Configures the 3D visualization settings for the simulation.
    
    Controls rendering of flock state, thermal conditions, wind fields, and other
    simulation elements for debugging, assessment, and analysis purposes. Most
    options can be toggled at runtime.
    """
    colors: ColorModel = Field(
        default_factory = ColorModel,
        description     = "Color configuration for visualization elements."
    )
    dark_mode: bool = Field(
        default     = True,
        description = "Whether to use a dark theme with black background."
    )
    glyphs: GlyphModel = Field(
        default_factory = GlyphModel,
        description     = "Glyph configuration for agent rendering."
    )
    grids: GridModel = Field(
        default_factory = GridModel,
        description     = "Grid configuration for field sampling."
    )
    opacity: OpacityModel = Field(
        default_factory = OpacityModel,
        description     = "Opacity configuration for visualization elements."
    )
    show_agents: bool = Field(
        default     = True,
        description = "Whether to render the flock agents as 3D glyphs."
    )
    show_graph: bool = Field(
        default     = True,
        description = "Whether to visualize the flock's communication graph connectivity."
    )
    show_safety: bool = Field(
        default     = True,
        description = "Whether to visualize the thermal safety boundary (T_max isosurface)."
    )
    show_thermal: bool = Field(
        default     = True,
        description = "Whether to color agents based on their sensed temperature."
    )
    show_trails: bool = Field(
        default     = True,
        description = "Whether to show motion trails behind agents."
    )
    show_wind: bool = Field(
        default     = True,
        description = "Whether to visualize the wind field with vector glyphs."
    )
    window_size: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1024, 768),
        description = "Size of the visualization window in pixels (width, height)."
    )
    window_title: str = Field(
        default     = "Thermur Simulation",
        description = "Title displayed in the visualization window frame."
    )
