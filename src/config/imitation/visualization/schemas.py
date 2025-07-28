"""
Visualization domain schemas for Pydantic validation.

This module provides configuration models for the 3D visualization system,
organized by functional areas for clarity while removing rarely-configured
parameters that can be hardcoded.
"""
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Annotated, Literal

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
RGBColor  = tuple[UnitFloat, UnitFloat, UnitFloat]


class ColorModel(BaseModel, extra="forbid"):
    """
    Color configuration for visualization elements.
    
    Controls the primary visual aesthetics of the simulation display.
    """
    agent_default: RGBColor = Field(
        default     = (0.2, 0.2, 0.8),
        description = (
            "Default RGB color for agents when thermal coloring is disabled, "
            "providing consistent visual identification of individual drones."
        )
    )
    colormap: Literal["plasma", "inferno", "viridis", "magma", "turbo"] = Field(
        default     = "plasma",
        description = (
            "Matplotlib colormap for temperature field visualization, mapping scalar "
            "values to colors for intuitive thermal gradient display."
        )
    )


class DisplayModel(BaseModel, extra="forbid"):
    """
    Display settings and element toggles.
    
    Controls which visual elements are rendered and basic window properties.
    """
    dark_mode: bool = Field(
        default     = True,
        description = (
            "Enable dark theme with black background."
        )
    )
    show_agents: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of agent positions as 3D glyphs in the visualization "
            "window for tracking individual drone states."
        )
    )
    show_graph: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of communication links between agents within range, "
            "visualizing the dynamic network topology of the flock."
        )
    )
    show_safety: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of thermal safety boundary isosurface at T_max to "
            "visualize Control Barrier Function constraint regions."
        )
    )
    show_thermal: bool = Field(
        default     = True,
        description = (
            "Toggle temperature-based agent coloring using the selected colormap to "
            "visualize thermal exposure across the flock."
        )
    )
    show_temperature_volume: bool = Field(
        default     = False,
        description = (
            "Toggle volumetric rendering of the temperature field to visualize thermal "
            "structures like updrafts, downdrafts, and temperature gradients."
        )
    )
    show_trails: bool = Field(
        default     = False,
        description = (
            "Toggle rendering of fading motion trails behind agents to visualize "
            "recent trajectories and emergent movement patterns."
        )
    )
    show_wind: bool = Field(
        default     = False,
        description = (
            "Toggle rendering of wind velocity vectors as arrow glyphs to visualize "
            "environmental forces affecting drone dynamics."
        )
    )
    window_size: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1024, 768),
        description = (
            "Render window dimensions in pixels (width, height)."
        )
    )


class GlyphModel(BaseModel, extra="forbid"):
    """
    Agent glyph rendering parameters.
    
    Controls the 3D representation of agents in the visualization.
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
            "Number of historical timesteps to include in motion trails, creating "
            "fading paths that reveal recent agent trajectories."
        )
    )
    type: Literal["sphere", "arrow"] = Field(
        default     = "sphere",
        description = (
            "3D geometry type for agent representation with sphere showing position "
            "only and arrow indicating both position and velocity direction."
        )
    )


class GridModel(BaseModel, extra="forbid"):
    """
    Sampling grid parameters for field visualization.
    
    Controls the resolution of continuous field discretization.
    """
    padding: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Buffer distance in meters added to flock bounding box for grid "
            "generation, preventing edge artifacts in volume rendering."
        )
    )
    temperature_resolution: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default     = (20, 20, 20),
        description = (
            "Voxel grid dimensions for temperature field interpolation, balancing "
            "visual smoothness against memory usage and rendering speed."
        )
    )
    wind_resolution: PositiveInt = Field(
        default     = 5,
        description = (
            "Grid points per dimension for wind vector visualization, creating a "
            "regular 3D lattice of arrow glyphs showing airflow."
        )
    )


class OpacityModel(BaseModel, extra="forbid"):
    """
    Transparency settings for visual elements.
    
    Controls the opacity of different visualization components.
    """
    agents: UnitFloat = Field(
        default     = 1.0,
        description = (
            "Alpha transparency value for agent glyphs where 1.0 is fully opaque "
            "for maximum visibility and lower values enable layered displays."
        )
    )
    graph: UnitFloat = Field(
        default     = 0.5,
        description = (
            "Alpha transparency value for communication edges, typically semi-transparent "
            "to avoid obscuring agents while showing network connectivity."
        )
    )
    temperature_volume: UnitFloat = Field(
        default     = 0.7,
        description = (
            "Alpha transparency value for volumetric temperature rendering, balanced to "
            "show thermal structures without completely obscuring the agents."
        )
    )