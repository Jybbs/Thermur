"""
Visualization configuration stores using hydra-zen.

This module provides store-based configurations for visualization components
using hydra-zen's decorator pattern. Each component is registered as a separate
build that can be referenced and overridden independently via Hydra's CLI.
"""
from .schemas                        import *
from config.utils.zen                import store, build
from thermur.imitation.visualization import Renderer, Sampler, Visualizer

visualization = store(group="visualization")
color         = ColorModel()
display       = DisplayModel()
glyph         = GlyphModel()
grid          = GridModel()
opacity       = OpacityModel()


@visualization(name="renderer")
def renderer_build():
    """
    Builder for visualization rendering component.
    
    Pre-builds the renderer with visual configuration parameters
    for efficient rendering of simulation elements. This component
    manages the visual appearance of agents, fields, and graphs.
    """
    return build(
        Renderer,
        colors    = color,
        glyphs    = glyph,
        opacities = opacity
    )

@visualization(name="sampler")
def sampler_build():
    """
    Builder for spatial grid sampling component.
    
    Pre-builds the grid sampler with fixed configuration parameters
    for efficient sampling of simulation data into visualization grids.
    This component handles discretization of continuous fields like
    temperature and wind for 3D rendering.
    """
    return build(
        Sampler,
        grid = grid
    )

@visualization(name="visualizer")
def visualizer_build():
    """
    Builder for 3D visualization system.
    
    Creates a real-time renderer for the thermal flock simulation that
    displays agent positions, communication topology, and temperature
    fields. Supports both interactive viewing and frame capture for
    offline video generation.
    
    The visualizer uses PyVista for efficient 3D rendering and supports
    various visual elements including agent glyphs, communication graphs,
    temperature fields, and safety boundaries.
    
    This builder now uses pre-built components to reduce initialization
    complexity and improve startup performance.
    """
    return build(
        Visualizer,
        colors       = color,
        display      = display,
        glyphs       = glyph,
        grids        = grid,
        opacity      = opacity,
        renderer     = "${visualization.renderer}",
        sampler      = "${visualization.sampler}",
        simulation   = "${simulation.env}"
    )