"""
Spatial sampling utilities for visualization data sources.

This module provides methods for sampling and discretizing continuous simulation
data from the environment for visualization purposes. It creates grid-based 
representations of thermal fields, wind vectors, and other spatially-distributed
data needed for 3D rendering.
"""
import numpy   as np
import pyvista as pv
import torch

from pyvista import PolyData, ImageData, UniformGrid
from torch   import Tensor
from typing  import Any, Union

# Sampling constants
DEFAULT_GRID_RESOLUTION = (20, 20, 20)  # Default grid dimensions for temperature field
DEFAULT_WIND_RESOLUTION = 5             # Default grid dimensions for wind field
DEFAULT_GRID_PADDING = 2.0              # Default padding around swarm bounding box
DEFAULT_AXIS_SCALE = 1.0                # Default scale for coordinate axes
DEFAULT_AXIS_COLORS = (                 # Default RGB colors for XYZ axes
    (1.0, 0.0, 0.0),  # Red for X
    (0.0, 1.0, 0.0),  # Green for Y
    (0.0, 0.0, 1.0)   # Blue for Z
)
MIN_DISTANCE = 1e-6                     # Minimum distance for normalization


def compute_grid_bounds(
    position : Tensor,
    padding  : float = DEFAULT_GRID_PADDING,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box for a grid based on agent positions.
    
    Calculates the minimum and maximum coordinates for a bounding box around
    the agent positions, with optional padding to ensure the entire simulation
    domain of interest is captured for visualization.
    
    Args:
        position : Tensor [N, 3] of agent positions
        padding  : Extra space around the swarm bounding box
        
    Returns:
        Tuple of (min_bounds, max_bounds) as numpy arrays
    """
    positions_np = position.detach().cpu().numpy()
    min_bounds = np.min(positions_np, axis=0) - padding
    max_bounds = np.max(positions_np, axis=0) + padding
    
    return min_bounds, max_bounds


def create_coordinate_axes(
    labels : tuple[str, str, str] = ("X", "Y", "Z"),
    origin : tuple[float, float, float] = (0, 0, 0),
    scale  : float = DEFAULT_AXIS_SCALE,
    colors : tuple[tuple[float, float, float]] = None,
) -> Any:
    """
    Create coordinate axes for orientation reference.
    
    Creates XYZ coordinate axes for the visualization to provide spatial 
    reference and orientation. The axes are centered at the specified origin
    and scaled according to the scale parameter. This helps orient viewers
    in 3D space and provides a sense of scale to the visualization.
    
    Args:
        labels : Labels for each axis
        origin : Origin point for the axes
        scale  : Size of the axes
        colors : Colors for each axis (X, Y, Z), defaults to RED, GREEN, BLUE
        
    Returns:
        PyVista axes object for the coordinate reference
    """
    axis_colors = DEFAULT_AXIS_COLORS if colors is None else colors
    
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
    
    # Set colors and labels
    axes.x_axis_color = axis_colors[0]
    axes.y_axis_color = axis_colors[1]
    axes.z_axis_color = axis_colors[2]
    
    axes.x_label = labels[0]
    axes.y_label = labels[1]
    axes.z_label = labels[2]
    
    return axes


def create_temperature_grid(
    environment    : Any,
    position       : Tensor,
    grid_resolution: tuple[int, int, int] = DEFAULT_GRID_RESOLUTION,
    padding        : float = DEFAULT_GRID_PADDING,
) -> Union[ImageData, UniformGrid]:
    """
    Create a uniform grid of temperature values from the environment.
    
    Samples the temperature field from the environment data source at regular
    grid points within a bounding box around the swarm. The resulting UniformGrid
    contains temperature data suitable for volume rendering or isosurface extraction.
    This grid represents the thermal environment that the swarm navigates through.
    
    Args:
        environment     : The simulation environment with data_source
        position        : Tensor [N, 3] of agent positions
        grid_resolution : Tuple of (nx, ny, nz) for grid resolution
        padding         : Extra space around the swarm bounding box
        
    Returns:
        PyVista UniformGrid with temperature scalar field data
    """
    # Get bounding box
    min_bounds, max_bounds = compute_grid_bounds(position, padding)
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    
    # Create a uniform grid
    grid = pv.ImageData(
        dimensions = grid_resolution,
        spacing    = ((x_max - x_min) / (grid_resolution[0] - 1),
                      (y_max - y_min) / (grid_resolution[1] - 1),
                      (z_max - z_min) / (grid_resolution[2] - 1)),
        origin     = (x_min, y_min, z_min)
    )
    
    # Sample temperature data at grid points
    grid_points = grid.points
    
    if hasattr(environment.data_source, 'query_thermal'):
        # Get temperature data from environment
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
        temp_values, _ = environment.data_source.query_thermal(grid_points_tensor)
        temp_np = temp_values.detach().cpu().numpy().flatten()
    else:
        # Generate synthetic temperature field if not available
        positions_np = position.detach().cpu().numpy()
        center = np.mean(positions_np, axis=0)
        distances = np.linalg.norm(grid_points - center, axis=1)
        temp_np = 100.0 * np.exp(-0.1 * distances)
    
    # Add temperature data to the grid
    grid.point_data["temperature"] = temp_np
    
    return grid


def create_wind_grid(
    environment     : Any,
    position        : Tensor,
    grid_resolution : int = DEFAULT_WIND_RESOLUTION,
    padding         : float = DEFAULT_GRID_PADDING,
) -> PolyData:
    """
    Create a grid of wind vectors from the environment data source.
    
    Samples the wind field from the environment at regular intervals within
    a bounding box around the swarm. The resulting PolyData contains points
    and vector data suitable for glyph-based visualization of the wind field.
    This helps visualize the air currents that the swarm interacts with during flight.
    
    Args:
        environment     : The simulation environment with data_source
        position        : Tensor [N, 3] of agent positions
        grid_resolution : Number of grid points in each dimension
        padding         : Extra space around the swarm bounding box
        
    Returns:
        PyVista PolyData with wind vector field data
    """
    # Get bounding box
    min_bounds, max_bounds = compute_grid_bounds(position, padding)
    x_min, y_min, z_min = min_bounds
    x_max, y_max, z_max = max_bounds
    
    # Create a regular grid of points
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    z = np.linspace(z_min, z_max, grid_resolution)
    
    # Create meshgrid and flatten to points
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    grid = pv.PolyData(points)
    
    # Sample wind data at grid points
    if hasattr(environment.data_source, 'query_wind'):
        # Get wind data from environment
        grid_points = torch.tensor(points, dtype=torch.float32)
        wind_vectors = environment.data_source.query_wind(grid_points)
        wind_np = wind_vectors.detach().cpu().numpy()
    else:
        # Generate synthetic wind field if not available
        positions_np = position.detach().cpu().numpy()
        center = np.mean(positions_np, axis=0)
        directions = points - center
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Avoid division by zero
        safe_distances = np.maximum(distances, MIN_DISTANCE)
        
        # Create normalized directions
        directions = directions / safe_distances
        
        # Create circular wind field
        wind_np = np.cross(directions, np.array([0, 0, 1]))
        
        # Scale magnitude with distance from center
        magnitude = 0.5 * np.exp(-0.1 * distances)
        wind_np = wind_np * magnitude
    
    # Add wind vectors to the grid
    grid.point_data["wind_velocity"] = wind_np
    
    return grid
