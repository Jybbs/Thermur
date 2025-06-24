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
    
    The visualization system provides real-time feedback during simulation and
    training, rendering the swarm's state, thermal conditions, wind fields, and
    other relevant data. This model configures all aspects of the visualization,
    from basic display properties to specialized rendering options for different
    simulation elements.
    
    The visualization serves several key purposes:
    1. Debugging and verification of swarm behavior
    2. Qualitative assessment of control strategies and emergent behaviors
    3. Visualization of thermal conditions and safety constraints
    4. Analysis of communication patterns and topology
    
    Most visualization options can be toggled on/off at runtime, making it easy
    to focus on specific aspects of the simulation as needed.
    """
    colormap: str = Field(
        default     = "plasma",
        description = ("The colormap used for thermal visualization. Controls the "
                       "color gradient used to represent temperature values throughout "
                       "the visualization. Common options include 'plasma', 'inferno', "
                       "and 'viridis', each offering different perceptual properties.")
    )
    dark_mode: bool = Field(
        default     = True,
        description = ("Whether to use a dark theme for visualization. Dark mode uses "
                       "a black background with high-contrast UI elements, which is "
                       "generally better for visualizing thermal data and reducing eye "
                       "strain during extended analysis sessions.")
    )
    enabled: bool = Field(
        default     = False,
        description = ("Master switch to enable/disable visualization during "
                       "training or simulation. When disabled, no visualization "
                       "window will be created, allowing for faster headless "
                       "execution on remote servers or during large batch runs.")
    )
    glyph_size: float = Field(
        default     = 0.15,
        description = ("Size scaling factor for agent glyphs. Larger values make "
                       "agents more visible but may cause visual clutter in dense "
                       "swarms. Adjust based on the typical spatial distribution "
                       "of your swarm configuration.")
    )
    glyph_type: str = Field(
        default     = "sphere",
        description = ("Type of glyph used to represent agents. Options include "
                       "'sphere' (position only) or 'arrow' (position and orientation). "
                       "Arrows are particularly useful for visualizing the alignment "
                       "and velocity directions within the swarm.")
    )
    grid_padding: float = Field(
        default     = 2.0,
        description = ("Extra space (in simulation units) added around the swarm's "
                       "bounding box when generating visualization grids. Larger "
                       "values provide more context around the swarm but may reduce "
                       "detail in the region of interest.")
    )
    show_agents: bool = Field(
        default     = True,
        description = ("Whether to render the swarm agents as 3D glyphs. This is the "
                       "primary visualization element showing the physical arrangement "
                       "of the swarm in space.")
    )
    show_graph: bool = Field(
        default     = False,
        description = ("Whether to visualize the swarm's communication graph. When "
                       "enabled, lines are drawn between agents that are connected "
                       "in the graph neural network, revealing the topology of "
                       "information flow within the swarm.")
    )
    show_safety: bool = Field(
        default     = False,
        description = ("Whether to visualize the thermal safety boundary (T_max). "
                       "When enabled, a semi-transparent isosurface is rendered at "
                       "the temperature threshold, showing regions that agents should "
                       "avoid to maintain thermal safety.")
    )
    show_thermal: bool = Field(
        default     = True,
        description = ("Whether to color agents based on their sensed temperature. "
                       "When enabled, agents are colored using the specified colormap "
                       "according to their current temperature, providing an immediate "
                       "visual indication of thermal conditions across the swarm.")
    )
    show_trails: bool = Field(
        default     = False,
        description = ("Whether to show motion trails behind agents. When enabled, "
                       "each agent leaves a fading trail showing its recent path, "
                       "which helps visualize the swarm's motion patterns and the "
                       "smoothness of trajectories.")
    )
    show_wind: bool = Field(
        default     = False,
        description = ("Whether to visualize the wind field with vector glyphs. "
                       "When enabled, arrows are placed on a 3D grid showing the "
                       "direction and magnitude of wind at various points, providing "
                       "context for understanding the external forces acting on the swarm.")
    )
    temp_grid_resolution: tuple = Field(
        default     = (20, 20, 20),
        description = ("Resolution of the temperature field sampling grid in each "
                       "dimension (nx, ny, nz). Higher values provide more detailed "
                       "thermal visualizations but increase computational cost. For "
                       "complex thermal fields with fine detail, use higher resolutions.")
    )
    wind_grid_resolution: int = Field(
        default     = 5,
        description = ("Resolution of the wind field sampling grid in each dimension. "
                       "Controls the density of arrow glyphs used to visualize the wind "
                       "field. Higher values show more detail but can create visual clutter.")
    )
    window_size: tuple = Field(
        default     = (1024, 768),
        description = ("Size of the visualization window in pixels (width, height). "
                       "Larger windows provide more detail but require more screen "
                       "space and computational resources for rendering.")
    )
    window_title: str = Field(
        default     = "Thermur Simulation",
        description = ("Title displayed in the window frame of the visualization. "
                       "Useful for identifying different simulation runs when "
                       "multiple windows are open.")
    )
