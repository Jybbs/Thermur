"""
Rendering functions for the Thermur visualization system.

This module provides specialized functions for rendering the different elements 
of the simulation. Each function creates PyVista actors that represent a 
particular aspect of the simulation state, such as agent positions and 
orientations, wind vectors, temperature fields, safety boundaries, and 
communication networks.
"""
import numpy   as np
import pyvista as pv
import torch

from pyvista import Plotter, PolyData
from torch   import Tensor
from typing  import Optional


def render_agents(
    colormap    : Optional[dict]   = None,
    glyph_size  : float            = 0.15,
    glyph_type  : str              = "sphere",
    plotter     : Plotter          = None,
    position    : Tensor           = None,
    show_trails : bool             = False,
    temperature : Optional[Tensor] = None,
    velocity    : Optional[Tensor] = None,
) -> list:
    """
    Render agents as glyphs (spheres or arrows) in the visualization.
    
    This function creates glyph actors for each agent in the swarm, with optional
    coloring based on temperature and orientation based on velocity.
    
    Args:
        plotter     : PyVista Plotter instance to render to
        position    : Tensor [N, 3] of agent positions
        velocity    : Optional Tensor [N, 3] of agent velocities
        temperature : Optional Tensor [N] of agent temperatures
        colormap    : Optional dictionary mapping temperature to color
        glyph_type  : Type of glyph to use ("sphere" or "arrow")
        glyph_size  : Size of the agent glyphs
        show_trails : Whether to show motion trails behind agents
        
    Returns:
        List of PyVista actors created for the agents
    """
    # Convert tensors to numpy arrays for PyVista
    positions_np = position.detach().cpu().numpy()
    
    # Create point cloud from agent positions
    cloud = pv.PolyData(positions_np)
    
    # Add velocity vectors as point cloud data if provided
    if velocity is not None:
        velocities_np = velocity.detach().cpu().numpy()
        cloud.point_data["velocity"] = velocities_np
    
    # Add temperature data for coloring if provided
    if temperature is not None:
        temps_np = temperature.detach().cpu().numpy()
        cloud.point_data["temperature"] = temps_np.flatten()
    
    # Create appropriate glyph based on type
    actors = []
    
    if glyph_type == "sphere":
        # Create sphere glyphs
        sphere = pv.Sphere(radius=glyph_size, phi_resolution=15, theta_resolution=15)
        
        # Render the glyphs
        if temperature is not None and colormap is not None:
            # Color glyphs by temperature
            min_temp = np.min(temps_np)
            max_temp = np.max(temps_np)
            
            # Get colors from colormap
            cmap_name = colormap.get("name", "plasma")
            scalars   = "temperature"
            
            glyph_actor = plotter.add_mesh(
                cloud.glyph(
                    geom      = sphere,
                    scale=False,
                ),
                scalars       = scalars,
                cmap          = cmap_name,
                clim          = (min_temp, max_temp),
                render_points_as_spheres = True,
            )
        else:
            # Use default color
            glyph_actor = plotter.add_mesh(
                cloud.glyph(
                    geom      = sphere,
                    scale=False,
                ),
                color = (0.2, 0.2, 0.8),
            )
        
        actors.append(glyph_actor)
    
    elif glyph_type == "arrow":
        # Create arrow glyphs oriented by velocity
        if velocity is not None:
            # Normalize velocity for orientation
            velocities_np = velocity.detach().cpu().numpy()
            magnitude = np.linalg.norm(velocities_np, axis=1, keepdims=True)
            magnitude = np.where(magnitude > 0, magnitude, 1.0)  # Avoid division by zero
            directions = velocities_np / magnitude
            
            # Add direction data to point cloud
            cloud.point_data["direction"] = directions
            
            # Create arrow glyphs
            arrow = pv.Arrow(
                shaft_radius       = glyph_size * 0.1,
                tip_length         = glyph_size * 0.5,
                tip_radius         = glyph_size * 0.2,
                shaft_resolution   = 10,
                tip_resolution     = 10,
            )
            
            # Render the glyphs
            if temperature is not None and colormap is not None:
                # Color glyphs by temperature
                min_temp = np.min(temps_np)
                max_temp = np.max(temps_np)
                
                # Get colors from colormap
                cmap_name = colormap.get("name", "plasma")
                scalars   = "temperature"
                
                glyph_actor = plotter.add_mesh(
                    cloud.glyph(
                        geom      = arrow,
                        orient    = "direction",
                        scale=False,
                    ),
                    scalars = scalars,
                    cmap    = cmap_name,
                    clim    = (min_temp, max_temp),
                )
            else:
                # Use default color
                glyph_actor = plotter.add_mesh(
                    cloud.glyph(
                        geom      = arrow,
                        orient    = "direction",
                        scale=False,
                    ),
                    color = (0.2, 0.2, 0.8),
                )
            
            actors.append(glyph_actor)
    
    # Add trails if requested
    if show_trails and velocity is not None:
        # Create trail lines based on position and velocity
        for i in range(len(positions_np)):
            pos = positions_np[i]
            vel = velocities_np[i]
            
            # Create trail points going backward from current position
            trail_length = 5
            trail_points = np.zeros((trail_length, 3))
            
            for j in range(trail_length):
                # Scale factor decreases for points further in the "past"
                scale = (trail_length - j) / trail_length
                trail_points[j] = pos - vel * j * 0.1 * scale
            
            # Create line from trail points
            trail = pv.Line(trail_points[0], trail_points[-1], resolution=trail_length)
            
            # Add trail with opacity based on temperature if available
            if temperature is not None:
                opacity = 0.3 + 0.7 * (temps_np[i] - min_temp) / (max_temp - min_temp) if max_temp > min_temp else 0.5
            else:
                opacity = 0.5
                
            trail_actor = plotter.add_mesh(
                trail,
                color         = (0.8, 0.8, 0.8),
                opacity       = opacity,
                line_width    = 2,
                render_lines_as_tubes = True,
            )
            
            actors.append(trail_actor)
    
    return actors


def render_temperature_field(
    colormap     : str     = "plasma",
    max_temp     : float   = None,
    min_temp     : float   = None,
    opacity      : float   = 0.5,
    plotter      : Plotter = None,
    temp_grid    : PolyData = None,
) -> list:
    """
    Render a 3D temperature field visualization.
    
    This function creates a volume rendering of a temperature field, showing
    thermal gradients throughout the simulation space.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        temp_grid  : PyVista PolyData or UniformGrid with temperature data
        min_temp   : Minimum temperature for colormap scaling
        max_temp   : Maximum temperature for colormap scaling
        colormap   : Name of colormap to use
        opacity    : Opacity of the volume rendering
        
    Returns:
        List of PyVista actors created for the temperature field
    """
    actors = []
    
    # Set up temperature range for colormap
    if min_temp is None:
        min_temp = temp_grid.get_data_range("temperature")[0]
    
    if max_temp is None:
        max_temp = temp_grid.get_data_range("temperature")[1]
    
    # Add the temperature volume with a custom opacity transfer function
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
    arrow_size  : float   = 0.1,
    color       : tuple   = (0.7, 0.7, 0.7),
    opacity     : float   = 0.8,
    plotter     : Plotter = None,
    scale       : float   = 0.1,
    wind_grid   : PolyData = None,
) -> list:
    """
    Render the wind field as arrow glyphs on a 3D grid.
    
    This function visualizes the wind field using arrows that indicate
    direction and magnitude of the wind at various points in the simulation.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        wind_grid  : PyVista PolyData or UniformGrid with wind vector data
        scale      : Scaling factor for arrow size based on wind magnitude
        color      : Color of the wind arrows
        opacity    : Opacity of the wind arrows
        arrow_size : Base size of the arrow glyphs
        
    Returns:
        List of PyVista actors created for the wind field
    """
    actors = []
    
    # Create arrow glyphs for the wind vectors
    arrow = pv.Arrow(
        shaft_radius       = arrow_size * 0.1,
        tip_length         = arrow_size * 0.5,
        tip_radius         = arrow_size * 0.2,
        shaft_resolution   = 10,
        tip_resolution     = 10,
    )
    
    # Normalize the wind vectors for orientation
    wind_grid["wind_direction"] = wind_grid["wind_velocity"].copy()
    
    # Calculate wind magnitude for scaling
    wind_grid["wind_magnitude"] = np.linalg.norm(wind_grid["wind_velocity"], axis=1)
    max_magnitude = np.max(wind_grid["wind_magnitude"])
    
    # Only show arrows where wind magnitude is significant
    threshold = 0.1 * max_magnitude
    mask = wind_grid["wind_magnitude"] > threshold
    masked_grid = wind_grid.extract_points(mask)
    
    if masked_grid.n_points > 0:
        # Add the wind field arrows
        glyph_actor = plotter.add_mesh(
            masked_grid.glyph(
                geom             = arrow,
                orient           = "wind_direction",
                factor           = scale,
                scale            = "wind_magnitude",
                scale_mode       = "scalar",
                rng              = [0, max_magnitude],
            ),
            color     = color,
            opacity   = opacity,
        )
        
        actors.append(glyph_actor)
    
    return actors


def render_safety_boundary(
    color            : tuple   = (0.9, 0.3, 0.3),
    margin           : float   = 10.0,
    max_temperature  : float   = None,
    opacity          : float   = 0.3,
    plotter          : Plotter = None,
    position         : Tensor  = None,
    temperature      : Tensor  = None,
    temperature_grad : Tensor  = None,
) -> list:
    """
    Render a safety boundary visualization showing the T_max isotherm.
    
    This function creates a semi-transparent isosurface representing the
    boundary where temperature equals T_max, which is the safety limit
    for the swarm.
    
    Args:
        plotter          : PyVista Plotter instance to render to
        position         : Tensor [N, 3] of agent positions
        temperature      : Tensor [N] of agent temperatures
        temperature_grad : Tensor [N, 3] of temperature gradients
        max_temperature  : Maximum safe temperature (T_max)
        margin           : Temperature margin for visualization
        color            : Color of the safety boundary
        opacity          : Opacity of the safety boundary
        
    Returns:
        List of PyVista actors created for the safety boundary
    """
    actors = []
    
    # Convert tensors to numpy arrays
    positions_np   = position.detach().cpu().numpy()
    temps_np       = temperature.detach().cpu().numpy().flatten()
    temp_grads_np  = temperature_grad.detach().cpu().numpy()
    
    # Extract bounding box from agent positions
    x_min, y_min, z_min = np.min(positions_np, axis=0) - 5.0
    x_max, y_max, z_max = np.max(positions_np, axis=0) + 5.0
    
    # Create a regular grid for the bounding box
    grid_size = 20
    grid = pv.UniformGrid(
        dimensions = (grid_size, grid_size, grid_size),
        spacing    = ((x_max - x_min) / (grid_size - 1),
                      (y_max - y_min) / (grid_size - 1),
                      (z_max - z_min) / (grid_size - 1)),
        origin     = (x_min, y_min, z_min)
    )
    
    # Create temperature field from agent measurements
    # We'll use inverse distance weighting for interpolation
    grid_points = grid.points
    grid_temps  = np.zeros(grid_points.shape[0])
    
    for i, point in enumerate(grid_points):
        # Calculate distances to all agents
        distances = np.linalg.norm(positions_np - point, axis=1)
        
        # Avoid division by zero
        distances = np.where(distances < 0.001, 0.001, distances)
        
        # Inverse distance weights
        weights = 1.0 / distances**2
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Interpolate temperature
        grid_temps[i] = np.sum(temps_np * weights)
    
    # Add temperature data to grid
    grid.point_data["temperature"] = grid_temps
    
    # Create isosurface at max_temperature
    contour = grid.contour([max_temperature - margin])
    
    if contour.n_points > 0:
        # Add safety boundary as isosurface
        boundary_actor = plotter.add_mesh(
            contour,
            color     = color,
            opacity   = opacity,
            smooth    = True,
        )
        
        actors.append(boundary_actor)
    
    return actors


def render_communication_graph(
    color      : tuple   = (0.7, 0.7, 0.9),
    edge_index : Tensor  = None,
    line_width : int     = 2,
    opacity    : float   = 0.5,
    plotter    : Plotter = None,
    position   : Tensor  = None,
) -> list:
    """
    Render the communication graph as lines between connected agents.
    
    This function visualizes the graph connectivity of the swarm by drawing
    lines between agents that are within communication range.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        position   : Tensor [N, 3] of agent positions
        edge_index : Tensor [2, E] connectivity graph
        color      : Color of the communication lines
        opacity    : Opacity of the communication lines
        line_width : Width of the communication lines
        
    Returns:
        List of PyVista actors created for the communication graph
    """
    actors = []
    
    # Convert tensors to numpy arrays
    positions_np = position.detach().cpu().numpy()
    
    # Check if there are any edges
    if edge_index.numel() > 0:
        # Extract source and target indices
        source_idx = edge_index[0].detach().cpu().numpy()
        target_idx = edge_index[1].detach().cpu().numpy()
        
        # Create a PolyData for the graph edges
        edges = pv.PolyData()
        
        # Add lines for each edge
        for i in range(len(source_idx)):
            source_pos = positions_np[source_idx[i]]
            target_pos = positions_np[target_idx[i]]
            
            # Create line from source to target
            line = pv.Line(source_pos, target_pos)
            
            # Add line to the plotter
            line_actor = plotter.add_mesh(
                line,
                color         = color,
                opacity       = opacity,
                line_width    = line_width,
                render_lines_as_tubes = True,
            )
            
            actors.append(line_actor)
    
    return actors
