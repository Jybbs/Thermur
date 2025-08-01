"""
Visualization domain builds for hydra-zen configuration.

This module provides pre-built components for 3D visualization using PyVista:

Rendering Components:
- Plotter       : PyVista's rendering window with configurable display properties,
                  camera controls, and multi-viewport support.
- Renderer      : Manages scene composition, lighting, and camera positioning.
- Visualizer    : Coordinates sampling and rendering to produce smooth animations
                  of simulation rollouts.

Sampling & Data Processing:
- Sampler       : Extracts visualization data from the simulation environment
                  including agent positions, orientations, wind vectors, and
                  fire perimeters.

Visual Elements:
- Arrow         : Visualizes both agents and wind vectors with size and color
                  mapping based on magnitude.
- Themes        : Pre-configured visual themes (dark/light) for different viewing
                  conditions.
"""
from .schemas                        import *
from hydra_zen                       import builds, make_config
from pyvista                         import Arrow, Plotter, themes
from thermur.imitation.visualization import Renderer, Sampler, Visualizer


VISUALIZATION_USER_CONFIG = make_config(
    vista = VistaModel()
)

VISUALIZATION_SYSTEM_BUILDS = {
    "agent_glyph": builds(
        Arrow,
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "plotter": builds(
        Plotter,
        window_size             = "${visualization.vista.window_size}",
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "renderer": builds(
        Renderer,
        agent_glyph             = "${_system.agent_glyph}",
        scalar_bar              = {},  # Default scalar bar config
        vista                   = "${visualization.vista}",
        wind_glyph              = "${_system.wind_glyph}",
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "sampler": builds(
        Sampler,
        grid_padding            = "${visualization.vista.grid_padding}",
        temperature_resolution  = "${visualization.vista.temperature_resolution}",
        wind_resolution         = "${visualization.vista.wind_resolution}",
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "theme_dark": builds(
        themes.DarkTheme,
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "theme_light": builds(
        themes.DocumentTheme,
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "visualizer": builds(
        Visualizer,
        plotter                 = "${_system.plotter}",
        renderer                = "${_system.renderer}",
        sampler                 = "${_system.sampler}",
        simulation              = "${_system.env}",
        vista                   = "${visualization.vista}",
        zen_partial             = True,
        populate_full_signature = True
    ),
    
    "wind_glyph": builds(
        Arrow,
        zen_partial             = True,
        populate_full_signature = True
    )
}