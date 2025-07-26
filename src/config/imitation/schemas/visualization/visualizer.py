"""
Visualization rendering configuration.

This module provides models for configuring rendering aspects of the 3D
visualization, including glyph representations, opacity settings, colors,
and display properties.
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
        description = (
            "Default RGB color triplet for agent glyphs when thermal coloring is "
            "disabled, using normalized values in range [0.0, 1.0]."
        )
    )
    colormap: Literal["plasma", "inferno", "viridis", "magma", "cividis", "turbo", "thermal"] = Field(
        default     = "plasma",
        description = (
            "Matplotlib-compatible colormap name for temperature field visualization, "
            "mapping scalar temperatures to RGB colors for intuitive display."
        )
    )
    graph_default: RGBColor = Field(
        default     = (0.7, 0.7, 0.9),
        description = (
            "Default RGB color triplet for rendering communication topology edges "
            "between agents within communication range R_comm."
        )
    )
    safety_default: RGBColor = Field(
        default     = (0.9, 0.3, 0.3),
        description = (
            "Default RGB color triplet for thermal safety boundary isosurface at "
            "T_max, typically red to indicate danger zones."
        )
    )
    scalar_bar_position_x: UnitFloat = Field(
        default     = 0.88,
        description = (
            "Normalized horizontal position [0,1] for temperature colorbar placement "
            "in render window, where 0 is left edge and 1 is right edge."
        )
    )
    scalar_bar_position_y: UnitFloat = Field(
        default     = 0.25,
        description = (
            "Normalized vertical position [0,1] for temperature colorbar placement "
            "in render window, where 0 is bottom edge and 1 is top edge."
        )
    )
    scalar_bar_title: str = Field(
        default     = "Temperature",
        description = (
            "Label text displayed above the temperature colorbar legend, typically "
            "including units for clarity in scientific visualizations."
        )
    )
    trail_default: RGBColor = Field(
        default     = (0.8, 0.8, 0.8),
        description = (
            "Default RGB color triplet for trajectory trail visualization showing "
            "agent motion history over recent timesteps for pattern analysis."
        )
    )
    wind_default: RGBColor = Field(
        default     = (0.7, 0.7, 0.7),
        description = (
            "Default RGB color triplet for wind vector glyph visualization, typically "
            "using neutral gray to avoid confusion with temperature colors."
        )
    )


class DisplayModel(BaseModel, extra="forbid"):
    """
    Configures display settings and toggles for visualization elements.
    
    These settings control the overall appearance and which elements
    are shown in the 3D visualization window.
    """
    dark_mode: bool = Field(
        default     = True,
        description = (
            "Enable dark theme with black background for reduced eye strain during "
            "extended visualization sessions and better contrast with thermal colors."
        )
    )
    show_agents: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of agent positions as 3D glyphs (spheres or arrows) "
            "in the visualization window for tracking individual drone states."
        )
    )
    show_graph: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of inter-agent communication links as edges when "
            "agents are within communication range R_comm of each other."
        )
    )
    show_safety: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of thermal safety boundary as semi-transparent isosurface "
            "at temperature T_max to visualize Control Barrier Function constraints."
        )
    )
    show_thermal: bool = Field(
        default     = True,
        description = (
            "Toggle temperature-based agent coloring using the selected colormap to "
            "visualize thermal exposure across the flock in real-time."
        )
    )
    show_trails: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of fading motion trails behind agents to visualize "
            "recent trajectories and emergent movement patterns in the flock."
        )
    )
    show_wind: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of wind velocity vectors as arrow glyphs to visualize "
            "environmental forces affecting drone flight dynamics."
        )
    )
    window_size: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1024, 768),
        description = (
            "PyVista render window dimensions in pixels as (width, height) tuple, "
            "affecting display resolution and computational rendering load."
        )
    )
    window_title: str = Field(
        default     = "Thermur Simulation",
        description = (
            "Window title text displayed in the operating system window manager, "
            "useful for distinguishing multiple concurrent simulation visualizations."
        )
    )


class GlyphModel(BaseModel, extra="forbid"):
    """
    Configures glyph rendering parameters for agent visualization.
    
    Glyphs are the 3D objects used to represent agents in the visualization.
    These settings control their appearance and behavior.
    """
    arrow_scale: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Multiplicative scale factor for arrow glyph size proportional to velocity "
            "magnitude, enhancing visual distinction of fast-moving agents."
        )
    )
    size: PositiveFloat = Field(
        default     = 0.15,
        description = (
            "Uniform scale factor in meters for agent glyph dimensions, balancing "
            "visibility against visual clutter in dense flocks."
        )
    )
    trail_length: NonNegativeInt = Field(
        default     = 5,
        description = (
            "Trail history length in timesteps for motion visualization, creating "
            "fading paths that reveal recent agent trajectories."
        )
    )
    type: Literal["sphere", "arrow"] = Field(
        default     = "sphere",
        description = (
            "3D geometry type for agent representation: 'sphere' for position-only "
            "display or 'arrow' for position and velocity direction."
        )
    )


class OpacityModel(BaseModel, extra="forbid"):
    """
    Configures opacity values for different visualization elements.
    
    Opacity values range from 0.0 (fully transparent) to 1.0 (fully opaque).
    These settings allow fine-tuning the visual layering of simulation elements.
    """
    agents: UnitFloat = Field(
        default     = 1.0,
        description = (
            "Alpha transparency value [0,1] for agent glyphs, with 1.0 fully opaque "
            "for maximum visibility and lower values for layered displays."
        )
    )
    graph: UnitFloat = Field(
        default     = 0.5,
        description = (
            "Alpha transparency value [0,1] for communication edges, typically semi-"
            "transparent to avoid obscuring agents while showing connectivity."
        )
    )
    safety: UnitFloat = Field(
        default     = 0.3,
        description = (
            "Alpha transparency value [0,1] for T_max isosurface, balancing visibility "
            "of danger zones against obstruction of agent views."
        )
    )
    trails: UnitFloat = Field(
        default     = 0.5,
        description = (
            "Alpha transparency value [0,1] for trajectory trails, creating ghost-like "
            "paths that fade with age to show motion history."
        )
    )
    wind: UnitFloat = Field(
        default     = 0.8,
        description = (
            "Alpha transparency value [0,1] for wind vector glyphs, allowing "
            "environmental force visualization without dominating the display."
        )
    )