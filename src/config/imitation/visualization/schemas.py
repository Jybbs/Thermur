"""
Visualization domain schemas for Pydantic validation.

This module consolidates all visualization configuration models including
rendering settings, display options, and sampling parameters.
"""
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
from typing   import Annotated, Literal

UnitFloat = Annotated[float, Field(ge=0.0, le=1.0)]
RGBColor  = tuple[UnitFloat, UnitFloat, UnitFloat]


class SamplingModel(BaseModel, extra="forbid"):
    """
    Unified sampling configuration for visualization.
    
    Controls how simulation data is sampled and processed for rendering,
    including grid resolution and temporal sampling rates.
    """
    output_dir: str = Field(
        default     = "visualization/frames",
        description = (
            "Directory path for saving rendered frames when save_frames is enabled."
        )
    )
    render_every_n_steps: PositiveInt = Field(
        default     = 10,
        description = (
            "Render frequency in simulation steps, balancing visual smoothness "
            "against computational cost."
        )
    )
    save_frames: bool = Field(
        default     = False,
        description = (
            "Enable saving rendered frames to disk for creating videos or "
            "publication figures."
        )
    )


class VisualizerModel(BaseModel, extra="forbid"):
    """
    Unified visualization configuration.
    
    Combines all visualization settings into a single model for configuring
    the 3D rendering system.
    """
    agent_color: RGBColor = Field(
        default     = (0.2, 0.2, 0.8),
        description = (
            "Default RGB color for agent glyphs."
        )
    )
    agent_opacity: UnitFloat = Field(
        default     = 1.0,
        description = (
            "Alpha transparency for agent glyphs."
        )
    )
    agent_size: PositiveFloat = Field(
        default     = 0.15,
        description = (
            "Scale factor in meters for agent glyphs."
        )
    )
    colormap: Literal["plasma", "inferno", "viridis", "magma", "turbo"] = Field(
        default     = "plasma",
        description = (
            "Colormap for temperature field visualization."
        )
    )
    dark_mode: bool = Field(
        default     = True,
        description = (
            "Enable dark theme with black background for better contrast."
        )
    )
    show_agents: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of agent positions."
        )
    )
    show_graph: bool = Field(
        default     = True,
        description = (
            "Toggle rendering of communication links."
        )
    )
    show_temperature: bool = Field(
        default     = True,
        description = (
            "Toggle temperature field visualization."
        )
    )
    window_size: tuple[PositiveInt, PositiveInt] = Field(
        default     = (1024, 768),
        description = (
            "PyVista render window dimensions in pixels as (width, height) tuple."
        )
    )
    window_title: str = Field(
        default     = "Thermur Simulation",
        description = (
            "Window title text displayed in the operating system window manager."
        )
    )