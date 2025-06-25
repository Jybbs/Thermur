"""
Spatial sampling utilities for visualization data sources.

This module provides methods for sampling and discretizing continuous simulation
data from the environment for visualization purposes. It creates grid-based 
representations of thermal fields, wind vectors, and other spatially-distributed
data needed for 3D rendering.

The sampling functions efficiently handle large-scale data by using vectorized
operations and leveraging PyVista's optimized data structures. The module
supports both real environment data sources and synthetic data generation
for testing purposes.
"""
import numpy   as np
import pyvista as pv
import torch

from configs.schemas.visualization import GridModel
from pyvista                       import PolyData, ImageData, UniformGrid
from thermur.simulation            import ThermalEnvironment
from torch                         import Tensor
from typing                        import Optional, Union


def compute_grid_bounds(
    position    : Tensor,
    grid_config : Optional[GridModel] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box for a grid based on agent positions.
    
    Calculates the minimum and maximum coordinates for a bounding box around
    the agent positions, with optional padding to ensure the entire simulation
    domain of interest is captured for visualization. The padding helps ensure
    that agents near the swarm boundary are properly visualized with their
    surrounding thermal and wind fields.
    
    Args:
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid parameters including padding
        
    Returns:
        Tuple of (min_bounds, max_bounds) as numpy arrays of shape [3]
    """
    padding = grid_config.padding if grid_config else 2.0
    
    agent_positions = position.detach().cpu().numpy()
    min_bounds = np.min(agent_positions, axis=0) - padding
    max_bounds = np.max(agent_positions, axis=0) + padding
    
    return min_bounds, max_bounds


def create_coordinate_axes(
    labels : tuple[str, str, str] = ("X", "Y", "Z"),
    origin : tuple[float, float, float] = (0, 0, 0),
    scale  : float = 1.0,
) -> pv.Axes:
    """
    Create coordinate axes for orientation reference.
    
    Creates XYZ coordinate axes for the visualization to provide spatial 
    reference and orientation. The axes are centered at the specified origin
    and scaled according to the scale parameter. This helps orient viewers
    in 3D space and provides a sense of scale to the visualization.
    
    The axes use a standard color scheme with red for X, green for Y, and
    blue for Z, following common conventions in 3D graphics. The arrow
    geometry is optimized for clarity at various viewing angles.
    
    Args:
        labels : Labels for each axis (X, Y, Z)
        origin : Origin point for the axes in 3D space
        scale  : Size scaling factor for the axes
        
    Returns:
        PyVista axes object configured for the coordinate reference
    """
    axis_colors = (
        (1.0, 0.0, 0.0),  # Red for X
        (0.0, 1.0, 0.0),  # Green for Y  
        (0.0, 0.0, 1.0),  # Blue for Z
    )
    
    axes = pv.Axes(
        actor_scale     = scale,
        cone_radius     = 0.2,
        cone_resolution = 10,
        label_size      = (0.1, 0.1, 0.1),
        line_width      = 2,
        shaft_length    = 0.8,
        shaft_radius    = 0.05,
        show_actor      = True,
        tip_length      = 0.2,
        tip_radius      = 0.1,
    )
    
    axes.x_axis_color = axis_colors[0]
    axes.y_axis_color = axis_colors[1]
    axes.z_axis_color = axis_colors[2]
    
    axes.x_label = labels[0]
    axes.y_label = labels[1]
    axes.z_label = labels[2]
    
    return axes


def create_temperature_grid(
    environment : Optional[ThermalEnvironment],
    position    : Tensor,
    grid_config : Optional[GridModel] = None,
) -> Union[ImageData, UniformGrid]:
    """
    Create a uniform grid of temperature values from the environment.
    
    Samples the temperature field from the environment data source at regular
    grid points within a bounding box around the swarm. The resulting UniformGrid
    contains temperature data suitable for volume rendering or isosurface extraction.
    This grid represents the thermal environment that the swarm navigates through.
    
    The function supports both real environment data sources and synthetic data
    generation for testing. When using real data, it efficiently queries the
    environment's thermal model. For synthetic data, it creates a radial
    temperature field centered on the swarm's centroid.
    
    Args:
        environment : The simulation environment with thermal data source
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid resolution and padding
        
    Returns:
        PyVista UniformGrid with temperature scalar field data
    """
    temperature_resolution = (
        grid_config.temperature_resolution if grid_config 
        else (20, 20, 20)
    )
    
    min_bounds, max_bounds = compute_grid_bounds(position, grid_config)
    
    grid_spacing = (
        (max_bounds[0] - min_bounds[0]) / (temperature_resolution[0] - 1),
        (max_bounds[1] - min_bounds[1]) / (temperature_resolution[1] - 1),
        (max_bounds[2] - min_bounds[2]) / (temperature_resolution[2] - 1)
    )
    
    grid = pv.ImageData(
        dimensions = temperature_resolution,
        spacing    = grid_spacing,
        origin     = min_bounds
    )
    
    grid_points = grid.points
    
    if (environment and hasattr(environment, 'data_source') and 
        hasattr(environment.data_source, 'query_thermal')):
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
        temperature_values, _ = environment.data_source.query_thermal(
            grid_points_tensor
        )
        temperature_array = temperature_values.detach().cpu().numpy().flatten()
    else:
        # Generate synthetic radial temperature field for testing
        agent_positions = position.detach().cpu().numpy()
        swarm_centroid = np.mean(agent_positions, axis=0)
        radial_distances = np.linalg.norm(grid_points - swarm_centroid, axis=1)
        
        # Exponential decay from center with base temperature
        base_temperature = 100.0
        decay_rate = 0.1
        temperature_array = base_temperature * np.exp(-decay_rate * radial_distances)
    
    grid.point_data["temperature"] = temperature_array
    
    return grid


def create_wind_grid(
    environment : Optional[ThermalEnvironment],
    position    : Tensor,
    grid_config : Optional[GridModel] = None,
) -> PolyData:
    """
    Create a grid of wind vectors from the environment data source.
    
    Samples the wind field from the environment at regular intervals within
    a bounding box around the swarm. The resulting PolyData contains points
    and vector data suitable for glyph-based visualization of the wind field.
    This helps visualize the air currents that the swarm interacts with during flight.
    
    The wind grid uses a coarser resolution than the temperature grid to reduce
    visual clutter while still conveying the overall flow patterns. The function
    supports both real wind data from the environment and synthetic circular
    flow patterns for testing purposes.
    
    Args:
        environment : The simulation environment with wind data source
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid resolution and padding
        
    Returns:
        PyVista PolyData with wind vector field data at each grid point
    """
    resolution = grid_config.wind_resolution if grid_config else 5
    
    min_bounds, max_bounds = compute_grid_bounds(position, grid_config)
    
    # Create evenly spaced grid coordinates
    x_coords = np.linspace(min_bounds[0], max_bounds[0], resolution)
    y_coords = np.linspace(min_bounds[1], max_bounds[1], resolution)
    z_coords = np.linspace(min_bounds[2], max_bounds[2], resolution)
    
    # Generate 3D grid points
    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords)
    grid_points = np.column_stack((
        x_grid.flatten(), 
        y_grid.flatten(), 
        z_grid.flatten()
    ))
    
    wind_grid = pv.PolyData(grid_points)
    
    if (environment and hasattr(environment, 'data_source') and 
        hasattr(environment.data_source, 'query_wind')):
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
        wind_vectors = environment.data_source.query_wind(grid_points_tensor)
        wind_array = wind_vectors.detach().cpu().numpy()
    else:
        # Generate synthetic circular wind field for testing
        agent_positions = position.detach().cpu().numpy()
        swarm_centroid = np.mean(agent_positions, axis=0)
        
        # Vector from center to each grid point
        radial_vectors = grid_points - swarm_centroid
        radial_distances = np.linalg.norm(radial_vectors, axis=1, keepdims=True)
        
        # Normalize radial vectors (with safety for zero distance)
        safe_distances = np.maximum(radial_distances, 1e-6)
        normalized_radial = radial_vectors / safe_distances
        
        # Create circular flow by crossing with vertical axis
        vertical_axis = np.array([0, 0, 1])
        circular_flow = np.cross(normalized_radial, vertical_axis)
        
        # Apply magnitude decay from center
        flow_magnitude = 0.5 * np.exp(-0.1 * radial_distances)
        wind_array = circular_flow * flow_magnitude
    
    wind_grid.point_data["wind_velocity"] = wind_array
    
    return wind_grid
