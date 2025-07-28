"""
Visualization configuration stores using hydra-zen.

This module provides store-based configurations for visualization components
using hydra-zen's decorator pattern. Each component is registered as a separate
build that can be referenced and overridden independently via Hydra's CLI.

The stores follow a flat structure where each component (plotter, renderer,
sampler, visualizer) is defined as a function decorated with @visualization(name=...).
This allows for clean interpolation references like ${visualization.plotter}
without nested builds, improving configuration clarity and override flexibility.
"""
from .schemas                        import VistaModel
from config.utils.zen                import build, store
from pyvista                         import Arrow, Plotter, Sphere, themes
from thermur.imitation.visualization import Renderer, Sampler, Visualizer

visualization = store(group="visualization")
vista         = VistaModel()


@visualization(name="agent_glyph")
def agent_glyph_build():
    """
    Builder for agent glyph geometry.
    
    Creates the appropriate glyph geometry based on the configured type.
    Returns either a sphere for position-only visualization or an arrow
    for position-and-velocity visualization. The geometry is reused across
    all agents for efficient rendering.
    """
    match vista.agent_type:
        case "sphere":
            return build(Sphere, radius=vista.agent_size)
        case "arrow" | _:
            return build(Arrow)

@visualization(name="plotter")
def plotter_build():
    """
    Builder for PyVista plotter window.
    
    Creates the main 3D rendering window with appropriate lighting, camera
    settings, and theme configuration. The plotter manages the visualization
    lifecycle and provides interactive controls for viewing the simulation.
    
    The theme is configured based on the dark_mode setting in VistaModel,
    and window dimensions are set according to the configuration.
    """
    return build(
        Plotter,
        lighting    = "three lights",
        off_screen  = False,
        theme       = "${visualization.theme}",
        title       = "Thermur Simulation",
        window_size = vista.window_size
    )

@visualization(name="renderer")
def renderer_build():
    """
    Builder for visualization rendering component.
    
    Pre-builds the renderer with visual configuration parameters for efficient
    rendering of simulation elements. This component manages the visual
    appearance of agents, communication graphs, temperature fields, wind
    vectors, and safety boundaries.
    
    The renderer uses cached glyph geometries and optimized rendering
    pipelines to maintain performance with large agent counts. All glyph
    parameters (size, type, scale) are embedded in the pre-built geometries.
    """
    return build(
        Renderer,
        agent_glyph = "${visualization.agent_glyph}",
        scalar_bar  = "${visualization.scalar_bar}",
        vista       = vista,
        wind_glyph  = "${visualization.wind_glyph}"
    )

@visualization(name="sampler")
def sampler_build():
    """
    Builder for spatial grid sampling component.
    
    Pre-builds the grid sampler with fixed configuration parameters for
    efficient sampling of simulation data into visualization grids. This
    component handles discretization of continuous fields like temperature
    and wind for 3D rendering.
    
    The sampler creates regular grids within dynamically-computed bounding
    boxes around the flock, ensuring complete coverage while minimizing
    unnecessary sampling overhead.
    """
    return build(
        Sampler,
        grid_padding           = vista.grid_padding,
        temperature_resolution = vista.temperature_resolution,
        wind_resolution        = vista.wind_resolution
    )

@visualization(name="scalar_bar")
def scalar_bar_build():
    """
    Builder for scalar bar configuration.
    
    Creates configuration for the temperature color bar legend that appears
    in volumetric temperature rendering. The scalar bar provides a visual
    reference for mapping colors to temperature values.
    """
    return build(
        dict,
        position_x = 0.88,
        position_y = 0.25,
        title      = "Temperature (°C)"
    )

@visualization(name="theme")
def theme_build():
    """
    Builder for PyVista visual theme.
    
    Configures the global theme for the visualization based on user preferences.
    Returns either a dark theme (black background) or document theme (white
    background) depending on the dark_mode setting in VistaModel.
    """
    if vista.dark_mode:
        return build(themes.DarkTheme)
    else:
        return build(themes.DocumentTheme)

@visualization(name="visualizer")
def visualizer_build():
    """
    Builder for 3D visualization system.
    
    Creates a real-time renderer for the thermal flock simulation that
    displays agent positions, communication topology, and temperature
    fields. Supports both interactive viewing and frame capture for
    offline video generation.
    
    The visualizer coordinates all visualization components and manages
    the rendering lifecycle. It uses pre-built components to reduce
    initialization complexity and improve startup performance.
    
    References:
    - ${visualization.plotter}: Main rendering window
    - ${visualization.renderer}: Element rendering manager
    - ${visualization.sampler}: Grid sampling utilities
    - ${simulation.env}: Simulation environment for data access
    """
    return build(
        Visualizer,
        plotter    = "${visualization.plotter}",
        renderer   = "${visualization.renderer}",
        sampler    = "${visualization.sampler}",
        simulation = "${simulation.env}",
        vista      = vista
    )

@visualization(name="wind_glyph")
def wind_glyph_build():
    """
    Builder for wind vector glyph geometry.
    
    Creates a reusable arrow geometry specifically for wind field visualization.
    Wind vectors are always displayed as arrows regardless of the agent glyph
    type, providing clear directional information about airflow patterns.
    """
    return build(Arrow)
