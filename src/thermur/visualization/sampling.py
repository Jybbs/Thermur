"""
Spatial sampling utilities for visualization data sources.

This module provides methods for sampling and discretizing continuous simulation
data from the environment for visualization purposes. It creates grid-based 
representations of thermal fields, wind vectors, and other spatially-distributed
data needed for 3D rendering.
"""
import numpy  as np
import pyvista as pv
import torch

from pyvista import PolyData
from torch   import Tensor
from typing  import Optional, Tuple, Any


def compute_grid_bounds(
    padding  : float = 2.0,
    position : Tensor = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box for a grid based on agent positions.
    
    Calculates the minimum and maximum coordinates for a bounding box around
    the agent positions, with optional padding to ensure the entire simulation
    domain of interest is captured for visualization.
    
    Args:
        padding  : Extra space around the swarm bounding box
        position : Tensor [N, 3] of agent positions
        
    Returns:
        Tuple of (min_bounds, max_bounds) as numpy arrays
    """
    # Extract positions as numpy array
    positions_np = position.detach().cpu().numpy()
    
    # Calculate bounding box with padding
    min_bounds = np.min(positions_np, axis=0) - padding
    max_bounds = np.max(positions_np, axis=0) + padding
    
    return min_bounds, max_bounds


def create_coordinate_axes(
    colors    : Tuple[Tuple[float, float, float]] = None,
    labels    : Tuple[str, str, str] = ("X", "Y", "Z"),
    origin    : Tuple[float, float, float] = (0, 0, 0),
    scale     : float = 1.0,
) -> list:
    """
    Create coordinate axes for orientation reference.
    
    Creates XYZ coordinate axes for the visualization to provide spatial 
    reference and orientation. The axes are centered at the specified origin
    and scaled according to the scale parameter.
    
    Args:
        colors : Colors for each axis (X, Y, Z)
        labels : Labels for each axis
        origin : Origin point for the axes
        scale  : Size of the axes
        
    Returns:
        PyVista axes object for the coordinate reference
    """
    if colors is None:
        colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1))  # RGB for XYZ
    
    # Create axes
    axes = pv.Axes(
        actor_scale    = scale,
        cone_radius    = 0.2,
        cone_resolution= 10,
        label_size     = (0.1, 0.1, 0.1),
        line_width     = 2,
        shaft_length   = 0.8,
        shaft_radius   = 0.05,
        show_actor     = True,
        tip_length     = 0.2,
        tip_radius     = 0.1,
    )
    
    # Set colors and labels
    axes.x_axis_color = colors[0]
    axes.y_axis_color = colors[1]
    axes.z_axis_color = colors[2]
    
    if labels:
        axes.x_label = labels[0]
        axes.y_label = labels[1]
        axes.z_label = labels[2]
    
    return axes


def create_temperature_grid(
    environment,
    position,
    grid_resolution : Tuple[int, int, int] = (20, 20, 20),
    padding         : float = 2.0,
) -> Any:
    """
    Create a uniform grid of temperature values from the environment.
    
    Samples the temperature field from the environment data source at regular
    grid points within a bounding box around the swarm. The resulting UniformGrid
    contains temperature data suitable for volume rendering or isosurface extraction.
    
    Args:
        environment     : The simulation environment with data_source
        position        : Tensor [N, 3] of agent positions
        grid_resolution : Tuple of (nx, ny, nz) for grid resolution
        padding         : Extra space around the swarm bounding box
        
    Returns:
        PyVista UniformGrid with temperature scalar field data
    """
    # Extract positions as numpy array
    positions_np = position.detach().cpu().numpy()
    
    # Calculate bounding box of agent positions with padding
    x_min, y_min, z_min = np.min(positions_np, axis=0) - padding
    x_max, y_max, z_max = np.max(positions_np, axis=0) + padding
    
    # Create a uniform grid
    grid = pv.ImageData(
        dimensions = grid_resolution,
        spacing    = ((x_max - x_min) / (grid_resolution[0] - 1),
                      (y_max - y_min) / (grid_resolution[1] - 1),
                      (z_max - z_min) / (grid_resolution[2] - 1)),
        origin     = (x_min, y_min, z_min)
    )
    
    # Get grid points
    grid_points = grid.points
    
    # Sample temperature data at these points
    if hasattr(environment.data_source, 'query_thermal'):
        # Convert points to torch tensor
        grid_points_tensor = torch.tensor(grid_points, dtype=torch.float32)
        
        # Query temperature data
        temp_values, _ = environment.data_source.query_thermal(grid_points_tensor)
        temp_np = temp_values.detach().cpu().numpy().flatten()
    else:
        # Create synthetic temperature data if not available
        center = np.mean(positions_np, axis=0)
        distances = np.linalg.norm(grid_points - center, axis=1)
        
        # Create temperature field decreasing with distance from center
        temp_np = 100.0 * np.exp(-0.1 * distances)
    
    # Add temperature data to the grid
    grid.point_data["temperature"] = temp_np
    
    return grid


def create_wind_grid(
    environment,
    position,
    grid_resolution : int = 5,
    padding         : float = 2.0,
) -> PolyData:
    """
    Create a grid of wind vectors from the environment data source.
    
    Samples the wind field from the environment at regular intervals within
    a bounding box around the swarm. The resulting PolyData contains points
    and vector data suitable for glyph-based visualization of the wind field.
    
    Args:
        environment     : The simulation environment with data_source
        position        : Tensor [N, 3] of agent positions
        grid_resolution : Number of grid points in each dimension
        padding         : Extra space around the swarm bounding box
        
    Returns:
        PyVista PolyData with wind vector field data
    """
    # Extract positions as numpy array
    positions_np = position.detach().cpu().numpy()
    
    # Calculate bounding box of agent positions with padding
    x_min, y_min, z_min = np.min(positions_np, axis=0) - padding
    x_max, y_max, z_max = np.max(positions_np, axis=0) + padding
    
    # Create a regular grid of points
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    z = np.linspace(z_min, z_max, grid_resolution)
    
    # Create meshgrid of coordinates
    X, Y, Z = np.meshgrid(x, y, z)
    
    # Flatten coordinates into a list of points
    points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    
    # Create PyVista PolyData for the points
    grid = pv.PolyData(points)
    
    # Sample wind data at these points
    if hasattr(environment.data_source, 'query_wind'):
        # Convert points to torch tensor
        grid_points = torch.tensor(points, dtype=torch.float32)
        
        # Query wind data
        wind_vectors = environment.data_source.query_wind(grid_points)
        wind_np = wind_vectors.detach().cpu().numpy()
    else:
        # Create synthetic wind data if not available
        center = np.mean(points, axis=0)
        directions = points - center
        distances = np.linalg.norm(directions, axis=1, keepdims=True)
        
        # Normalize directions
        with np.errstate(divide='ignore', invalid='ignore'):
            directions = np.divide(directions, distances, 
                               out=np.zeros_like(directions), 
                               where=distances!=0)
        
        # Create circular wind field
        wind_np = np.cross(directions, np.array([0, 0, 1]))
        
        # Scale magnitude with distance from center
        magnitude = 0.5 * np.exp(-0.1 * distances)
        wind_np = wind_np * magnitude
    
    # Add wind vectors to the grid
    grid.point_data["wind_velocity"] = wind_np
    
    return grid
