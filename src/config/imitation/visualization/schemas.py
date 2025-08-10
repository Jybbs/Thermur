"""
Visualization domain schemas for Pydantic validation.

This module provides a unified configuration model for the 3D visualization
system, consolidating all visual settings into a single comprehensive schema
for clarity and ease of configuration.
"""
from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt
from typing   import Annotated, Literal

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
RGBColor  = tuple[UnitFloat, UnitFloat, UnitFloat]


class VistaModel(BaseModel, extra="forbid"):
    """
    Unified configuration for the 3D visualization system.

    This comprehensive schema controls all aspects of the real-time simulation
    visualization including agent rendering, environmental field display, visual
    aesthetics, and interactive settings. Parameters are organized alphabetically
    with descriptive names for intuitive configuration.
    """
    agent_color: RGBColor = Field(
        default     = (0.2, 0.2, 0.8),
        description = (
            "Default RGB color for agents when thermal coloring is disabled, "
            "providing consistent visual identification of individual drones."
        )
    )
    agent_opacity: UnitFloat = Field(
        default     = 1.0,
        description = (
            "Alpha transparency value for agent glyphs where 1.0 is fully opaque "
            "for maximum visibility and lower values enable layered displays."
        )
    )
    arrow_scale: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Multiplicative scale factor for arrow glyph size proportional to velocity "
            "magnitude, enhancing visual distinction of fast-moving agents."
        )
    )
    auto_save_frames: bool = Field(
        default     = False,
        description = (
            "Automatically save visualization frames after each render for creating "
            "animations or documenting simulation results."
        )
    )
    colormap: Literal["plasma", "inferno", "viridis", "magma", "turbo"] = Field(
        default     = "plasma",
        description = (
            "Matplotlib colormap for temperature field visualization, mapping scalar "
            "values to colors for intuitive thermal gradient display."
        )
    )
    dark_mode: bool = Field(
        default     = True,
        description = (
            "Enable dark theme with black background for reduced eye strain."
        )
    )
    frame_output_dir: str = Field(
        default     = "data/frames",
        description = (
            "Directory path for saving visualization frames. Created automatically "
            "if it doesn't exist when frame capture is enabled."
        )
    )
    graph_opacity: UnitFloat = Field(
        default     = 0.5,
        description = (
            "Alpha transparency value for communication edges, typically semi-transparent "
            "to avoid obscuring agents while showing network connectivity."
        )
    )
    grid_padding: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Buffer distance in meters added to flock bounding box for grid "
            "generation, preventing edge artifacts in volume rendering."
        )
    )
    log_video: bool = Field(
        default     = True,
        description = (
            "Capture visualization frames to a buffer for logging videos to WandB. "
            "Independent of --watch flag which controls live window display."
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
    show_safety_boundary: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of thermal safety boundary isosurface at T_max to "
            "visualize Control Barrier Function constraint regions."
        )
    )
    show_temperature_volume: bool = Field(
        default     = False,
        description = (
            "Toggle volumetric rendering of the temperature field to visualize thermal "
            "structures like updrafts, downdrafts, and temperature gradients."
        )
    )
    show_thermal_colors: bool = Field(
        default     = True,
        description = (
            "Toggle temperature-based agent coloring using the selected colormap to "
            "visualize thermal exposure across the flock."
        )
    )
    show_trails: bool = Field(
        default     = False,
        description = (
            "Toggle rendering of fading motion trails behind agents to visualize "
            "recent trajectories and emergent movement patterns."
        )
    )
    show_wind_arrows: bool = Field(
        default     = False,
        description = (
            "Toggle rendering of wind velocity vectors as arrow glyphs to visualize "
            "environmental forces affecting drone dynamics."
        )
    )
    temperature_opacity: UnitFloat = Field(
        default     = 0.7,
        description = (
            "Alpha transparency value for volumetric temperature rendering, balanced to "
            "show thermal structures without completely obscuring the agents."
        )
    )
    temperature_resolution: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        default     = (20, 20, 20),
        description = (
            "Voxel grid dimensions for temperature field interpolation, balancing "
            "visual smoothness against memory usage and rendering speed."
        )
    )
    trail_length: NonNegativeInt = Field(
        default     = 5,
        description = (
            "Number of historical timesteps to include in motion trails, creating "
            "fading paths that reveal recent agent trajectories."
        )
    )
    trajectories_to_monitor: PositiveInt = Field(
        default     = 5,
        description = (
            "Maximum number of trajectory simulations to monitor from each training batch. "
            "Each trajectory represents a complete flock simulation and gets its own video "
            "stream in WandB for comparison."
        )
    )
    wind_resolution: PositiveInt = Field(
        default     = 5,
        description = (
            "Grid points per dimension for wind vector visualization, creating a "
            "regular 3D lattice of arrow glyphs showing airflow."
        )
    )
    window_size: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1024, 768),
        description = (
            "Render window dimensions in pixels (width, height)."
        )
    )
