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

from pyvista import Plotter, PolyData
from torch   import Tensor
from typing  import Any, Optional

# Rendering constants
DEFAULT_SPHERE_RESOLUTION = 15  # Resolution for sphere glyphs
DEFAULT_ARROW_RESOLUTION = 10   # Resolution for arrow glyphs
DEFAULT_AGENT_COLOR = (0.2, 0.2, 0.8)  # Default blue color for agents
DEFAULT_WIND_COLOR = (0.7, 0.7, 0.7)   # Default gray color for wind
DEFAULT_GRAPH_COLOR = (0.7, 0.7, 0.9)  # Default blue-gray for graph edges
DEFAULT_SAFETY_COLOR = (0.9, 0.3, 0.3) # Default red color for safety boundary
DEFAULT_TRAIL_COLOR = (0.8, 0.8, 0.8)  # Default white-gray for trails
TRAIL_LENGTH = 5                       # Number of points in motion trails
SAFETY_GRID_SIZE = 20                  # Resolution of safety boundary grid
SAFETY_PADDING = 5.0                   # Padding around safety boundary
WIND_THRESHOLD_FACTOR = 0.1            # Threshold factor for wind visualization


def render_agents(
    plotter     : Plotter,
    position    : Tensor,
    velocity    : Optional[Tensor] = None,
    temperature : Optional[Tensor] = None,
    colormap    : Optional[dict[str, Any]] = None,
    glyph_type  : str = "sphere",
    glyph_size  : float = 0.15,
    show_trails : bool = False
) -> list[Any]:
    """
    Render agents as glyphs (spheres or arrows) in the visualization.
    
    Creates visual representations of each agent in the swarm using either sphere
    glyphs (showing position only) or arrow glyphs (showing position and direction).
    Agents can be colored by temperature using the provided colormap, and optional
    motion trails can be drawn to show recent movement patterns.
    
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
    positions_np = position.detach().cpu().numpy()
    cloud = pv.PolyData(positions_np)
    actors = []
    
    # Add data to point cloud
    velocities_np = None
    temps_np = None
    
    if velocity is not None:
        velocities_np = velocity.detach().cpu().numpy()
        cloud.point_data["velocity"] = velocities_np
    
    if temperature is not None:
        temps_np = temperature.detach().cpu().numpy()
        cloud.point_data["temperature"] = temps_np.flatten()
    
    # Render based on glyph type
    if glyph_type == "sphere":
        sphere = pv.Sphere(
            radius=glyph_size, 
            phi_resolution=DEFAULT_SPHERE_RESOLUTION, 
            theta_resolution=DEFAULT_SPHERE_RESOLUTION
        )
        
        if temperature is not None and colormap is not None:
            min_temp = np.min(temps_np)
            max_temp = np.max(temps_np)
            cmap_name = colormap.get("name", "plasma")
            
            glyph_actor = plotter.add_mesh(
                cloud.glyph(geom=sphere, scale=False),
                scalars="temperature",
                cmap=cmap_name,
                clim=(min_temp, max_temp),
                render_points_as_spheres=True,
            )
        else:
            glyph_actor = plotter.add_mesh(
                cloud.glyph(geom=sphere, scale=False),
                color=DEFAULT_AGENT_COLOR,
            )
        
        actors.append(glyph_actor)
    
    elif glyph_type == "arrow" and velocity is not None:
        # Normalize velocity for orientation
        velocities_np = velocity.detach().cpu().numpy()
        magnitude = np.linalg.norm(velocities_np, axis=1, keepdims=True)
        magnitude = np.where(magnitude > 0, magnitude, 1.0)  # Avoid division by zero
        directions = velocities_np / magnitude
        
        # Add direction data to point cloud
        cloud.point_data["direction"] = directions
        
        # Create arrow glyphs
        arrow = pv.Arrow(
            shaft_radius     = glyph_size * 0.1,
            tip_length       = glyph_size * 0.5,
            tip_radius       = glyph_size * 0.2,
            shaft_resolution = DEFAULT_ARROW_RESOLUTION,
            tip_resolution   = DEFAULT_ARROW_RESOLUTION,
        )
        
        if temperature is not None and colormap is not None:
            min_temp = np.min(temps_np)
            max_temp = np.max(temps_np)
            cmap_name = colormap.get("name", "plasma")
            
            glyph_actor = plotter.add_mesh(
                cloud.glyph(geom=arrow, orient="direction", scale=False),
                scalars="temperature",
                cmap=cmap_name,
                clim=(min_temp, max_temp),
            )
        else:
            glyph_actor = plotter.add_mesh(
                cloud.glyph(geom=arrow, orient="direction", scale=False),
                color=DEFAULT_AGENT_COLOR,
            )
        
        actors.append(glyph_actor)
    
    # Add trails if requested
    if show_trails and velocity is not None and velocities_np is not None:
        trail_decay_factor = 0.1  # Controls trail length
        
        for i in range(len(positions_np)):
            agent_pos = positions_np[i]
            agent_vel = velocities_np[i]
            
            # Create trail points with decreasing intensity
            trail_points = np.zeros((TRAIL_LENGTH, 3))
            
            for j in range(TRAIL_LENGTH):
                decay = (TRAIL_LENGTH - j) / TRAIL_LENGTH
                trail_points[j] = agent_pos - agent_vel * j * trail_decay_factor * decay
            
            # Create line from trail points
            trail = pv.Line(
                trail_points[0], 
                trail_points[-1], 
                resolution=TRAIL_LENGTH
            )
            
            # Compute opacity based on temperature if available
            opacity = 0.5
            if temperature is not None and temps_np is not None:
                if np.max(temps_np) > np.min(temps_np):
                    temp_factor = (temps_np[i] - np.min(temps_np)) / (np.max(temps_np) - np.min(temps_np))
                    opacity = 0.3 + 0.7 * temp_factor
            
            trail_actor = plotter.add_mesh(
                trail,
                color=DEFAULT_TRAIL_COLOR,
                opacity=opacity,
                line_width=2,
                render_lines_as_tubes=True,
            )
            
            actors.append(trail_actor)
    
    return actors


def render_temperature_field(
    plotter    : Plotter,
    temp_grid  : PolyData,
    min_temp   : Optional[float] = None,
    max_temp   : Optional[float] = None,
    colormap   : str = "plasma",
    opacity    : float = 0.5
) -> list[Any]:
    """
    Render a 3D temperature field visualization.
    
    Creates a volume rendering of a temperature field, showing thermal gradients
    throughout the simulation space. This visualization provides insights into
    the thermal environment that the swarm navigates through, helping to understand
    the thermal currents, hot spots, and thermal gradients.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        temp_grid  : PyVista PolyData or UniformGrid with temperature data
        min_temp   : Minimum temperature for colormap scaling (auto-detected if None)
        max_temp   : Maximum temperature for colormap scaling (auto-detected if None)
        colormap   : Name of colormap to use
        opacity    : Opacity of the volume rendering
        
    Returns:
        List of PyVista actors created for the temperature field
    """
    actors = []
    
    # Auto-detect temperature range if not provided
    if min_temp is None or max_temp is None:
        temp_range = temp_grid.get_data_range("temperature")
        min_temp = temp_range[0] if min_temp is None else min_temp
        max_temp = temp_range[1] if max_temp is None else max_temp
    
    # Add volume visualization
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
    plotter    : Plotter,
    wind_grid  : PolyData,
    scale      : float = 0.1,
    arrow_size : float = 0.1,
    color      : tuple[float, float, float] = None,
    opacity    : float = 0.8
) -> list[Any]:
    """
    Render the wind field as arrow glyphs on a 3D grid.
    
    Visualizes wind vectors as directional arrows that show both orientation
    and magnitude of airflow. Arrows are placed on a grid with density based
    on the provided wind_grid. Only significant wind vectors are displayed to
    reduce visual clutter.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        wind_grid  : PyVista PolyData with wind vector data
        scale      : Scaling factor for arrow size based on wind magnitude
        arrow_size : Base size of the arrow glyphs
        color      : Color of the wind arrows (defaults to DEFAULT_WIND_COLOR)
        opacity    : Opacity of the wind arrows
        
    Returns:
        List of PyVista actors created for the wind field
    """
    actors = []
    wind_color = DEFAULT_WIND_COLOR if color is None else color
    
    # Create arrow glyphs for the wind vectors
    arrow = pv.Arrow(
        shaft_radius     = arrow_size * 0.1,
        tip_length       = arrow_size * 0.5,
        tip_radius       = arrow_size * 0.2,
        shaft_resolution = DEFAULT_ARROW_RESOLUTION,
        tip_resolution   = DEFAULT_ARROW_RESOLUTION,
    )
    
    # Prepare wind data for visualization
    wind_grid["wind_direction"] = wind_grid["wind_velocity"].copy()
    wind_grid["wind_magnitude"] = np.linalg.norm(wind_grid["wind_velocity"], axis=1)
    max_magnitude = np.max(wind_grid["wind_magnitude"])
    
    # Filter to show only significant wind vectors
    threshold = WIND_THRESHOLD_FACTOR * max_magnitude
    mask = wind_grid["wind_magnitude"] > threshold
    masked_grid = wind_grid.extract_points(mask)
    
    if masked_grid.n_points > 0:
        glyph_actor = plotter.add_mesh(
            masked_grid.glyph(
                geom       = arrow,
                orient     = "wind_direction",
                factor     = scale,
                scale      = "wind_magnitude",
                scale_mode = "scalar",
                rng        = [0, max_magnitude],
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
    margin           : float = 10.0,
    color            : tuple[float, float, float] = None,
    opacity          : float = 0.3
) -> list[Any]:
    """
    Render a safety boundary visualization showing the T_max isotherm.
    
    Creates a semi-transparent isosurface representing the boundary where
    temperature equals T_max, which is the safety limit for the swarm.
    This visualization helps to verify that the CBF (Control Barrier Function)
    is working correctly by showing the areas that agents should avoid.
    
    Args:
        plotter          : PyVista Plotter instance to render to
        position         : Tensor [N, 3] of agent positions
        temperature      : Tensor [N] of agent temperatures
        temperature_grad : Tensor [N, 3] of temperature gradients
        max_temperature  : Maximum safe temperature (T_max)
        margin           : Temperature margin for visualization
        color            : Color of the safety boundary (defaults to DEFAULT_SAFETY_COLOR)
        opacity          : Opacity of the safety boundary
        
    Returns:
        List of PyVista actors created for the safety boundary
    """
    actors = []
    safety_color = DEFAULT_SAFETY_COLOR if color is None else color
    
    # Convert tensors to numpy arrays
    positions_np  = position.detach().cpu().numpy()
    temps_np      = temperature.detach().cpu().numpy().flatten()
    
    # Extract bounding box from agent positions with padding
    min_bounds = np.min(positions_np, axis=0) - SAFETY_PADDING
    max_bounds = np.max(positions_np, axis=0) + SAFETY_PADDING
    
    # Create a regular grid for the bounding box
    grid = pv.UniformGrid(
        dimensions = (SAFETY_GRID_SIZE, SAFETY_GRID_SIZE, SAFETY_GRID_SIZE),
        spacing    = ((max_bounds[0] - min_bounds[0]) / (SAFETY_GRID_SIZE - 1),
                      (max_bounds[1] - min_bounds[1]) / (SAFETY_GRID_SIZE - 1),
                      (max_bounds[2] - min_bounds[2]) / (SAFETY_GRID_SIZE - 1)),
        origin     = min_bounds
    )
    
    # Interpolate temperature field using inverse distance weighting
    grid_points = grid.points
    grid_temps = np.zeros(grid_points.shape[0])
    
    # Vectorized calculation of distances between grid points and agent positions
    # This is much more efficient than looping through each grid point
    MIN_DISTANCE = 0.001  # Minimum distance to avoid division by zero
    
    for i, point in enumerate(grid_points):
        # Calculate all distances at once
        distances = np.linalg.norm(positions_np - point, axis=1)
        distances = np.maximum(distances, MIN_DISTANCE)
        
        # Inverse square distance weights
        weights = 1.0 / (distances * distances)
        weights_sum = np.sum(weights)
        
        # Weighted average of temperatures
        grid_temps[i] = np.sum(temps_np * (weights / weights_sum))
    
    # Add temperature data to grid
    grid.point_data["temperature"] = grid_temps
    
    # Create isosurface at max_temperature - margin
    contour = grid.contour([max_temperature - margin])
    
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
    plotter    : Plotter,
    position   : Tensor,
    edge_index : Tensor,
    color      : tuple[float, float, float] = None,
    opacity    : float = 0.5,
    line_width : int = 2
) -> list[Any]:
    """
    Render the communication graph as lines between connected agents.
    
    Visualizes the graph connectivity of the swarm by drawing lines between
    agents that are within communication range. This visualization reveals
    the topology of information flow within the swarm and helps debug
    connectivity issues or understand emergent behaviors based on communication
    patterns.
    
    Args:
        plotter    : PyVista Plotter instance to render to
        position   : Tensor [N, 3] of agent positions
        edge_index : Tensor [2, E] connectivity graph
        color      : Color of the communication lines (defaults to DEFAULT_GRAPH_COLOR)
        opacity    : Opacity of the communication lines
        line_width : Width of the communication lines
        
    Returns:
        List of PyVista actors created for the communication graph
    """
    actors = []
    graph_color = DEFAULT_GRAPH_COLOR if color is None else color
    
    # Convert tensors to numpy arrays
    positions_np = position.detach().cpu().numpy()
    
    # Check if there are any edges
    if edge_index.numel() > 0:
        # Extract source and target indices
        source_idx = edge_index[0].detach().cpu().numpy()
        target_idx = edge_index[1].detach().cpu().numpy()
        
        # Add lines for each edge in the graph
        for i in range(len(source_idx)):
            source_pos = positions_np[source_idx[i]]
            target_pos = positions_np[target_idx[i]]
            
            # Create line from source to target
            line = pv.Line(source_pos, target_pos)
            
            # Add line to the plotter
            line_actor = plotter.add_mesh(
                line,
                color               = graph_color,
                opacity             = opacity,
                line_width          = line_width,
                render_lines_as_tubes = True,
            )
            
            actors.append(line_actor)
    
    return actors
