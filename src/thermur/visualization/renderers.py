"""
Rendering functions for the Thermur visualization system.

This module provides specialized functions for rendering the different elements 
of the simulation. Each function creates PyVista actors that represent a 
particular aspect of the simulation state, such as agent positions and 
orientations, wind vectors, temperature fields, safety boundaries, and 
communication networks.

The rendering functions are designed to be efficient and scalable, using
vectorized operations where possible and PyVista's optimized rendering
pipeline. Each function returns a list of actors that can be managed by
the main visualizer for updates and cleanup.
"""
from __future__                import annotations
from pyvista                   import Actor, ImageData, PolyData, Plotter
from torch                     import Tensor
from typing                    import Optional
from configs.imitation.schemas import VisualizationModel

import numpy   as np
import pyvista as pv


def render_agents(
    plotter     : Plotter,
    position    : Tensor,
    velocity    : Optional[Tensor]             = None,
    temperature : Optional[Tensor]             = None,
    colormap    : Optional[str]                = None,
    config      : Optional[VisualizationModel] = None,
    show_trails : bool                         = False,
) -> list[Actor]:
    """
    Render agents as glyphs (spheres or arrows) in the visualization.
    
    Creates visual representations of each agent in the swarm using either sphere
    glyphs (showing position only) or arrow glyphs (showing position and direction).
    Agents can be colored by temperature using the provided colormap, and optional
    motion trails can be drawn to show recent movement patterns.
    
    The function optimizes rendering performance by using PyVista's efficient
    glyph operations and batching similar operations together. Temperature-based
    coloring uses the simulation's thermal data to provide immediate visual
    feedback about each agent's thermal state.
    
    Args:
        plotter     : PyVista Plotter instance to render to
        position    : Agent positions tensor of shape [N, 3]
        velocity    : Agent velocities tensor of shape [N, 3] (optional)
        temperature : Agent temperatures tensor of shape [N, 1] (optional)
        colormap    : Temperature colormap for thermal visualization
        glyphs      : Configuration for glyph appearance and behavior
        colors      : Configuration for color settings
        opacities   : Configuration for opacity values
        show_trails : Whether to render motion trails behind agents
        
    Returns:
        List of PyVista actors created for the agents
    """
    actors      = []
    positions   = position.detach().cpu().numpy()
    point_cloud = pv.PolyData(positions)
    
    if velocity is not None:
        velocities = velocity.detach().cpu().numpy()
        point_cloud["velocity"] = velocities
    
    if temperature is not None:
        temps = temperature.detach().cpu().numpy().flatten()
        point_cloud["temperature"] = temps
    
    glyph_geom = (
        pv.Sphere(radius=glyphs.size) 
        if glyphs.type == "sphere" 
        else pv.Arrow()
    )
    
    if glyphs.type == "arrow" and velocity is not None:
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        safe_norms = np.maximum(norms, 1e-6)
        point_cloud["direction"] = velocities / safe_norms
        
        agent_glyphs = point_cloud.glyph(
            geom   = glyph_geom, 
            orient = "direction",
            scale  = False
        )
        
    else:
        agent_glyphs = point_cloud.glyph(
            geom   = glyph_geom, 
            orient = False,
            scale  = False
        )
    
    mesh_params = {
        "render_points_as_spheres": glyphs.type == "sphere",
    }
    
    if temperature is not None and colormap:
        mesh_params.update(
            scalars = "temperature",
            cmap    = colormap,
            clim    = (temps.min(), temps.max()),
        )
    else:
        mesh_params["color"] = colors.agent_default
    
    actors.append(plotter.add_mesh(agent_glyphs, **mesh_params))
    
    if show_trails and velocity is not None:
        trail_actors = _render_trails(
            plotter     = plotter,
            positions   = positions,
            velocities  = velocities,
            temperature = temps if temperature is not None else None,
            colormap    = colormap,
            glyphs      = glyphs,
            colors      = colors,
            opacities   = opacities,
        )
        actors.extend(trail_actors)
    
    return actors


def _render_trails(
    plotter     : Plotter,
    positions   : np.ndarray,
    velocities  : np.ndarray,
    temperature : Optional[np.ndarray],
    colormap    : Optional[str],
    glyphs      : BaseModel,
    colors      : BaseModel,
    opacities   : BaseModel,
) -> list[Actor]:
    """
    Helper function to render agent motion trails.
    
    Creates fading trail lines behind each agent based on their velocity.
    Trails fade out with distance/time for visual clarity.
    
    Args:
        plotter     : PyVista Plotter instance
        positions   : Agent positions array [N, 3]
        velocities  : Agent velocities array [N, 3]
        temperature : Agent temperatures array [N] (optional)
        colormap    : Temperature colormap (optional)
        glyphs      : Glyph configuration
        colors      : Color configuration
        opacities   : Opacity configuration
        
    Returns:
        List of actors for the trails
    """
    n_agents = len(positions)
    n_points = glyphs.trail_length
    
    t          = np.linspace(0, 0.1 * n_points, n_points)
    decay      = np.linspace(1, 0, n_points)[:, None]
    offsets    = velocities[:, None] * t * decay
    points     = (positions[:, None] - offsets).reshape(-1, 3)
    indices    = np.arange(n_agents * n_points).reshape(n_agents, n_points)
    trail_mesh = pv.MultipleLines(points=points, lines=indices)
    
    if temperature is not None:
        trail_mesh["temperature"] = np.repeat(temperature, n_points)
    
    params = {
        "opacity"               : opacities.trails,
        "line_width"            : 2,
        "render_lines_as_tubes" : True,
    }
    
    if temperature is not None and colormap:
        params.update(
            scalars = "temperature",
            cmap    = colormap,
            clim    = (temperature.min(), temperature.max()),
        )
    else:
        params["color"] = colors.trail_default
    
    return [plotter.add_mesh(trail_mesh, **params)]


def render_temperature_field(
    plotter   : Plotter,
    temp_grid : ImageData,
    colors    : BaseModel,
    opacities : BaseModel,
    min_temp  : Optional[float] = None,
    max_temp  : Optional[float] = None,
) -> list[Actor]:
    """
    Render a 3D temperature field visualization.
    
    Creates a volume rendering of a temperature field, showing thermal gradients
    throughout the simulation space. This visualization provides insights into
    the thermal environment that the swarm navigates through, helping to understand
    the thermal currents, hot spots, and thermal gradients.
    
    The volume rendering uses PyVista's efficient GPU-accelerated rendering
    pipeline to display large temperature fields in real-time. The colormap
    and opacity settings can be configured to highlight specific temperature
    ranges or features of interest.
    
    Args:
        plotter   : PyVista Plotter instance to render to
        temp_grid : PyVista UniformGrid with temperature data
        colors    : Configuration for color mapping
        opacities : Configuration for opacity values
        min_temp  : Minimum temperature for colormap scaling (auto if None)
        max_temp  : Maximum temperature for colormap scaling (auto if None)
        
    Returns:
        List of PyVista actors created for the temperature field
    """
    bounds = temp_grid.get_data_range("temperature")
    
    return [
        plotter.add_volume(
            temp_grid,
            cmap            = colors.colormap,
            clim            = (min_temp or bounds[0], max_temp or bounds[1]),
            opacity         = "linear",
            scalar_bar_args = {
                "title"      : colors.scalar_bar_title,
                "position_x" : colors.scalar_bar_position_x,
                "position_y" : colors.scalar_bar_position_y,
            },
        )
    ]


def render_wind_field(
    plotter   : Plotter,
    wind_grid : PolyData,
    glyphs    : BaseModel,
    colors    : BaseModel,
    opacities : BaseModel,
) -> list[Actor]:
    """
    Render the wind field as arrow glyphs on a 3D grid.
    
    Visualizes wind vectors as directional arrows that show both orientation
    and magnitude of airflow. Arrows are placed on a grid with density based
    on the provided wind_grid. Only significant wind vectors are displayed to
    reduce visual clutter.
    
    The rendering optimizes performance by filtering out small wind vectors
    and using PyVista's efficient glyph operations. The arrow size and color
    can be configured to match the overall visualization style.
    
    Args:
        plotter   : PyVista Plotter instance to render to
        wind_grid : PyVista PolyData with wind vector data
        glyphs    : Configuration for glyph appearance
        colors    : Configuration for color settings
        opacities : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the wind field
    """
    velocities = wind_grid["wind_velocity"]
    magnitudes = np.linalg.norm(velocities, axis=1)
    wind_grid["wind_magnitude"] = magnitudes
    
    # Filter insignificant vectors to reduce clutter
    threshold = 0.1 * magnitudes.max()
    filtered_grid = wind_grid.threshold(
        value   = threshold, 
        scalars = "wind_magnitude"
    )
    
    if not filtered_grid.n_points:
        return []
    
    wind_glyphs = filtered_grid.glyph(
        geom   = pv.Arrow(),
        orient = "wind_velocity",
        scale  = "wind_magnitude",
        factor = glyphs.arrow_scale,
    )
    
    mesh_params = {
        "color"   : colors.wind_default,
        "opacity" : opacities.wind,
    }
    
    return [plotter.add_mesh(mesh=wind_glyphs, **mesh_params)]


def render_safety_boundary(
    plotter          : Plotter,
    position         : Tensor,
    temperature      : Tensor,
    temperature_grad : Tensor,
    max_temperature  : float,
    grids            : BaseModel,
    colors           : BaseModel,
    opacities        : BaseModel,
) -> list[Actor]:
    """
    Render a safety boundary visualization showing the T_max isotherm.
    
    Creates a semi-transparent isosurface representing the boundary where
    temperature equals T_max, which is the safety limit for the swarm.
    This visualization helps to verify that the CBF (Control Barrier Function)
    is working correctly by showing the areas that agents should avoid.
    
    The function uses PyVista's built-in interpolation for temperature field
    estimation and isosurface extraction for efficient rendering.
    
    Args:
        plotter          : PyVista Plotter instance to render to
        position         : Agent positions tensor of shape [N, 3]
        temperature      : Agent temperatures tensor of shape [N, 1]
        temperature_grad : Temperature gradients tensor of shape [N, 3]
        max_temperature  : Maximum safe temperature (T_max)
        grids            : Configuration for grid sampling parameters
        colors           : Configuration for color settings
        opacities        : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the safety boundary
    """
    positions   = position.cpu().numpy()
    point_cloud = pv.PolyData(positions)
    point_cloud["temperature"] = temperature.cpu().numpy().ravel()
    
    min_bounds = np.array([positions[:, i].min() - grids.padding for i in range(3)])
    max_bounds = np.array([positions[:, i].max() + grids.padding for i in range(3)])
    resolution = np.array(grids.temperature_resolution)
    
    target_grid = pv.ImageData(
        dimensions = grids.temperature_resolution,
        spacing    = (max_bounds - min_bounds) / (resolution - 1),
        origin     = min_bounds,
    )
    
    # Use sample method to interpolate temperature values
    grid = target_grid.sample(point_cloud)
    
    try:
        contour = grid.contour([max_temperature])
        if contour.n_points == 0:
            return []
    except:
        return []
    
    return [
        plotter.add_mesh(
            mesh    = contour.smooth(n_iter=50),
            color   = colors.safety_default,
            opacity = opacities.safety,
        )
    ]


def render_communication_graph(
    plotter   : Plotter,
    position  : Tensor,
    edge_index: Tensor,
    colors    : BaseModel,
    opacities : BaseModel,
) -> list[Actor]:
    """
    Render the communication graph as lines between connected agents.
    
    Visualizes the graph connectivity of the swarm by drawing lines between
    agents that are within communication range. This visualization reveals
    the topology of information flow within the swarm and helps debug
    connectivity issues or understand emergent behaviors based on communication
    patterns.
    
    The function efficiently handles large graphs by batching line creation
    and using PyVista's optimized rendering for line meshes. The visual
    style can be configured to match the overall visualization theme.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        position   : Agent positions tensor of shape [N, 3]
        edge_index : Connectivity graph tensor of shape [2, E]
        colors     : Configuration for color settings
        opacities  : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the communication graph
    """
    if not edge_index.numel():
        return []
    
    edges = edge_index.cpu().numpy()
    lines = np.column_stack(
        [
            np.full(edges.shape[1], 2),
            edges[0],
            edges[1]
        ]
    ).ravel()
    
    mesh = pv.PolyData(position.cpu().numpy())
    mesh.lines = lines
    
    return [
        plotter.add_mesh(
            mesh                   = mesh,
            color                  = colors.graph_default,
            opacity                = opacities.graph,
            line_width             = 2,
            render_lines_as_tubes  = True,
        )
    ]
