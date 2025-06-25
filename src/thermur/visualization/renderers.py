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
import numpy   as np
import pyvista as pv

from configs.schemas.visualization import (
    ColorModel, 
    GlyphModel, 
    GridModel,
    OpacityModel
)
from pyvista import Plotter, PolyData, UniformGrid
from torch   import Tensor
from typing  import Any, Optional


def render_agents(
    plotter      : Plotter,
    position     : Tensor,
    velocity     : Optional[Tensor] = None,
    temperature  : Optional[Tensor] = None,
    colormap     : Optional[Any] = None,
    glyph_config : Optional[GlyphModel] = None,
    color_config : Optional[ColorModel] = None,
    show_trails  : bool = False,
) -> list[Any]:
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
        plotter      : PyVista Plotter instance to render to
        position     : Agent positions tensor of shape [N, 3]
        velocity     : Agent velocities tensor of shape [N, 3] (optional)
        temperature  : Agent temperatures tensor of shape [N, 1] (optional)
        colormap     : Temperature colormap for thermal visualization
        glyph_config : Configuration for glyph appearance and behavior
        color_config : Configuration for color settings
        show_trails  : Whether to render motion trails behind agents
        
    Returns:
        List of PyVista actors created for the agents
    """
    glyph_type = glyph_config.type if glyph_config else "sphere"
    glyph_size = glyph_config.size if glyph_config else 0.15
    agent_color = color_config.agent_default if color_config else (0.2, 0.2, 0.8)
    
    agent_positions = position.detach().cpu().numpy()
    point_cloud = pv.PolyData(agent_positions)
    actors = []
    
    agent_velocities = None
    agent_temperatures = None
    
    if velocity is not None:
        agent_velocities = velocity.detach().cpu().numpy()
        point_cloud.point_data["velocity"] = agent_velocities
    
    if temperature is not None:
        agent_temperatures = temperature.detach().cpu().numpy()
        point_cloud.point_data["temperature"] = agent_temperatures.flatten()
    
    if glyph_type == "sphere":
        sphere_resolution = 15
        sphere = pv.Sphere(
            radius           = glyph_size, 
            phi_resolution   = sphere_resolution, 
            theta_resolution = sphere_resolution
        )
        
        if temperature is not None and colormap is not None:
            temperature_range = (
                np.min(agent_temperatures), 
                np.max(agent_temperatures)
            )
            
            glyph_actor = plotter.add_mesh(
                point_cloud.glyph(geom=sphere, scale=False),
                scalars                  = "temperature",
                cmap                     = colormap,
                clim                     = temperature_range,
                render_points_as_spheres = True,
            )
        else:
            glyph_actor = plotter.add_mesh(
                point_cloud.glyph(geom=sphere, scale=False),
                color = agent_color,
            )
        
        actors.append(glyph_actor)
    
    elif glyph_type == "arrow" and velocity is not None:
        velocity_magnitudes = np.linalg.norm(agent_velocities, axis=1, keepdims=True)
        safe_magnitudes = np.where(velocity_magnitudes > 0, velocity_magnitudes, 1.0)
        normalized_directions = agent_velocities / safe_magnitudes
        
        point_cloud.point_data["direction"] = normalized_directions
        
        arrow_resolution = 10
        arrow_scale = glyph_config.arrow_scale if glyph_config else 0.1
        arrow = pv.Arrow(
            shaft_radius     = glyph_size * 0.1,
            tip_length       = glyph_size * 0.5,
            tip_radius       = glyph_size * 0.2,
            shaft_resolution = arrow_resolution,
            tip_resolution   = arrow_resolution,
        )
        
        if temperature is not None and colormap is not None:
            temperature_range = (
                np.min(agent_temperatures), 
                np.max(agent_temperatures)
            )
            
            glyph_actor = plotter.add_mesh(
                point_cloud.glyph(geom=arrow, orient="direction", scale=False),
                scalars = "temperature",
                cmap    = colormap,
                clim    = temperature_range,
            )
        else:
            glyph_actor = plotter.add_mesh(
                point_cloud.glyph(geom=arrow, orient="direction", scale=False),
                color = agent_color,
            )
        
        actors.append(glyph_actor)
    
    if show_trails and velocity is not None and agent_velocities is not None:
        trail_length = glyph_config.trail_length if glyph_config else 5
        trail_decay_factor = 0.1
        trail_color = (0.8, 0.8, 0.8)
        
        for agent_idx in range(len(agent_positions)):
            current_position = agent_positions[agent_idx]
            current_velocity = agent_velocities[agent_idx]
            
            trail_points = np.zeros((trail_length, 3))
            
            for point_idx in range(trail_length):
                decay_weight = (trail_length - point_idx) / trail_length
                trail_offset = (
                    current_velocity * point_idx * 
                    trail_decay_factor * decay_weight
                )
                trail_points[point_idx] = current_position - trail_offset
            
            trail_line = pv.Line(
                trail_points[0], 
                trail_points[-1], 
                resolution=trail_length
            )
            
            trail_opacity = 0.5
            if temperature is not None and agent_temperatures is not None:
                temperature_range = (
                    np.max(agent_temperatures) - np.min(agent_temperatures)
                )
                if temperature_range > 0:
                    normalized_temp = (
                        (agent_temperatures[agent_idx] - np.min(agent_temperatures)) 
                        / temperature_range
                    )
                    trail_opacity = 0.3 + 0.7 * normalized_temp
            
            trail_actor = plotter.add_mesh(
                trail_line,
                color                 = trail_color,
                opacity               = trail_opacity,
                line_width            = 2,
                render_lines_as_tubes = True,
            )
            
            actors.append(trail_actor)
    
    return actors


def render_temperature_field(
    plotter        : Plotter,
    temp_grid      : UniformGrid,
    color_config   : Optional[ColorModel] = None,
    opacity_config : Optional[OpacityModel] = None,
    min_temp       : Optional[float] = None,
    max_temp       : Optional[float] = None,
) -> list[Any]:
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
        plotter        : PyVista Plotter instance to render to
        temp_grid      : PyVista UniformGrid with temperature data
        color_config   : Configuration for color mapping
        opacity_config : Configuration for opacity values
        min_temp       : Minimum temperature for colormap scaling (auto if None)
        max_temp       : Maximum temperature for colormap scaling (auto if None)
        
    Returns:
        List of PyVista actors created for the temperature field
    """
    actors = []
    
    colormap = color_config.colormap if color_config else "plasma"
    opacity = 0.5
    
    if min_temp is None or max_temp is None:
        temperature_bounds = temp_grid.get_data_range("temperature")
        min_temp = temperature_bounds[0] if min_temp is None else min_temp
        max_temp = temperature_bounds[1] if max_temp is None else max_temp
    
    volume_actor = plotter.add_volume(
        temp_grid,
        cmap       = colormap,
        clim       = (min_temp, max_temp),
        opacity    = opacity,
        scalar_bar = True,
        stitle     = "Temperature",
    )
    
    actors.append(volume_actor)
    
    return actors


def render_wind_field(
    plotter        : Plotter,
    wind_grid      : PolyData,
    glyph_config   : Optional[GlyphModel] = None,
    color_config   : Optional[ColorModel] = None,
    opacity_config : Optional[OpacityModel] = None,
) -> list[Any]:
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
        plotter        : PyVista Plotter instance to render to
        wind_grid      : PyVista PolyData with wind vector data
        glyph_config   : Configuration for glyph appearance
        color_config   : Configuration for color settings
        opacity_config : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the wind field
    """
    actors = []
    
    wind_color = color_config.wind_default if color_config else (0.7, 0.7, 0.7)
    opacity = opacity_config.wind if opacity_config else 0.8
    arrow_scale = glyph_config.arrow_scale if glyph_config else 0.1
    arrow_size = glyph_config.size if glyph_config else 0.1
    
    arrow_resolution = 10
    arrow = pv.Arrow(
        shaft_radius     = arrow_size * 0.1,
        tip_length       = arrow_size * 0.5,
        tip_radius       = arrow_size * 0.2,
        shaft_resolution = arrow_resolution,
        tip_resolution   = arrow_resolution,
    )
    
    wind_grid["wind_direction"] = wind_grid["wind_velocity"].copy()
    wind_magnitudes = np.linalg.norm(wind_grid["wind_velocity"], axis=1)
    wind_grid["wind_magnitude"] = wind_magnitudes
    max_wind_magnitude = np.max(wind_magnitudes)
    
    magnitude_threshold = 0.1 * max_wind_magnitude
    significant_wind_mask = wind_magnitudes > magnitude_threshold
    filtered_wind_grid = wind_grid.extract_points(significant_wind_mask)
    
    if filtered_wind_grid.n_points > 0:
        glyph_actor = plotter.add_mesh(
            filtered_wind_grid.glyph(
                geom       = arrow,
                orient     = "wind_direction",
                factor     = arrow_scale,
                scale      = "wind_magnitude",
                scale_mode = "scalar",
                rng        = [0, max_wind_magnitude],
            ),
            color   = wind_color,
            opacity = opacity,
        )
        
        actors.append(glyph_actor)
    
    return actors


def render_safety_boundary(
    plotter          : Plotter,
    position         : Tensor,
    temperature      : Tensor,
    temperature_grad : Tensor,
    max_temperature  : float,
    grid_config      : Optional[GridModel] = None,
    color_config     : Optional[ColorModel] = None,
    opacity_config   : Optional[OpacityModel] = None,
) -> list[Any]:
    """
    Render a safety boundary visualization showing the T_max isotherm.
    
    Creates a semi-transparent isosurface representing the boundary where
    temperature equals T_max, which is the safety limit for the swarm.
    This visualization helps to verify that the CBF (Control Barrier Function)
    is working correctly by showing the areas that agents should avoid.
    
    The function uses vectorized operations for efficient temperature field
    interpolation, avoiding loops where possible. The isosurface is smoothed
    for better visual quality and to clearly show the safety boundary shape.
    
    Args:
        plotter          : PyVista Plotter instance to render to
        position         : Agent positions tensor of shape [N, 3]
        temperature      : Agent temperatures tensor of shape [N, 1]
        temperature_grad : Temperature gradients tensor of shape [N, 3]
        max_temperature  : Maximum safe temperature (T_max)
        grid_config      : Configuration for grid sampling parameters
        color_config     : Configuration for color settings
        opacity_config   : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the safety boundary
    """
    actors = []
    
    safety_color = color_config.safety_default if color_config else (0.9, 0.3, 0.3)
    opacity = opacity_config.safety if opacity_config else 0.3
    
    grid_padding = grid_config.padding if grid_config else 2.0
    temperature_resolution = (
        grid_config.temperature_resolution if grid_config else (20, 20, 20)
    )
    
    # Convert tensors to numpy arrays
    agent_positions = position.detach().cpu().numpy()
    agent_temperatures = temperature.detach().cpu().numpy().flatten()
    
    # Extract bounding box from agent positions with padding
    min_bounds = np.min(agent_positions, axis=0) - grid_padding
    max_bounds = np.max(agent_positions, axis=0) + grid_padding
    
    # Create a regular grid for the bounding box
    grid = pv.UniformGrid(
        dimensions = temperature_resolution,
        spacing    = (
            (max_bounds[0] - min_bounds[0]) / (temperature_resolution[0] - 1),
            (max_bounds[1] - min_bounds[1]) / (temperature_resolution[1] - 1),
            (max_bounds[2] - min_bounds[2]) / (temperature_resolution[2] - 1)
        ),
        origin     = min_bounds
    )
    
    # Vectorized temperature interpolation using broadcasting
    grid_points = grid.points
    n_grid_points = grid_points.shape[0]
    n_agents = agent_positions.shape[0]
    
    # Reshape for broadcasting: grid_points (n_grid, 1, 3), positions (1, n_agents, 3)
    grid_expanded = grid_points.reshape(n_grid_points, 1, 3)
    positions_expanded = agent_positions.reshape(1, n_agents, 3)
    
    # Compute all pairwise distances at once
    distances = np.linalg.norm(grid_expanded - positions_expanded, axis=2)
    distances = np.maximum(distances, 0.001)  # Avoid division by zero
    
    # Inverse distance weighting
    weights = 1.0 / (distances * distances)
    normalized_weights = weights / weights.sum(axis=1, keepdims=True)
    
    # Weighted temperature interpolation
    grid_temperatures = (normalized_weights * agent_temperatures).sum(axis=1)
    
    # Add temperature data to grid
    grid.point_data["temperature"] = grid_temperatures
    
    # Create isosurface at max_temperature with small margin
    temperature_margin = 1.0
    contour = grid.contour([max_temperature - temperature_margin])
    
    if contour.n_points > 0:
        boundary_actor = plotter.add_mesh(
            contour,
            color   = safety_color,
            opacity = opacity,
            smooth  = True,
        )
        
        actors.append(boundary_actor)
    
    return actors


def render_communication_graph(
    plotter        : Plotter,
    position       : Tensor,
    edge_index     : Tensor,
    color_config   : Optional[ColorModel] = None,
    opacity_config : Optional[OpacityModel] = None,
) -> list[Any]:
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
        plotter        : PyVista Plotter instance to render to
        position       : Agent positions tensor of shape [N, 3]
        edge_index     : Connectivity graph tensor of shape [2, E]
        color_config   : Configuration for color settings
        opacity_config : Configuration for opacity values
        
    Returns:
        List of PyVista actors created for the communication graph
    """
    actors = []
    
    graph_color = color_config.graph_default if color_config else (0.7, 0.7, 0.9)
    opacity = opacity_config.graph if opacity_config else 0.5
    line_width = 2
    
    # Convert tensors to numpy arrays
    agent_positions = position.detach().cpu().numpy()
    
    # Check if there are any edges
    if edge_index.numel() > 0:
        source_indices = edge_index[0].detach().cpu().numpy()
        target_indices = edge_index[1].detach().cpu().numpy()
        
        # Create all lines at once for better performance
        lines = []
        for source_idx, target_idx in zip(source_indices, target_indices):
            source_position = agent_positions[source_idx]
            target_position = agent_positions[target_idx]
            
            line = pv.Line(source_position, target_position)
            lines.append(line)
        
        # Merge all lines into a single mesh for efficient rendering
        if lines:
            merged_lines = lines[0]
            for line in lines[1:]:
                merged_lines = merged_lines.merge(line)
            
            # Add merged lines to the plotter
            graph_actor = plotter.add_mesh(
                merged_lines,
                color                 = graph_color,
                opacity               = opacity,
                line_width            = line_width,
                render_lines_as_tubes = True,
            )
            
            actors.append(graph_actor)
    
    return actors
