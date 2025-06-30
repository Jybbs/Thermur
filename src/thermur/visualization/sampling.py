"""
Spatial sampling utilities for visualization data sources.

This module provides methods for sampling and discretizing continuous simulation
data from the environment for visualization purposes. It creates grid-based 
representations of thermal fields, wind vectors, and other spatially-distributed
data needed for 3D rendering.

The sampling functions efficiently handle large-scale data by using vectorized
operations and leveraging PyVista's optimized data structures.
"""
from __future__ import annotations
from pydantic   import BaseModel
from pyvista    import Axes, ImageData, PolyData
from torch      import Tensor
from typing     import Any

import numpy   as np
import pyvista as pv
import torch


def compute_grid_bounds(
    position    : Tensor,
    grid_config : BaseModel,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the bounding box for a grid based on agent positions.
    
    Calculates the minimum and maximum coordinates for a bounding box around
    the agent positions, with padding to ensure the entire simulation
    domain of interest is captured for visualization.
    
    Args:
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid parameters including padding
        
    Returns:
        Tuple of (min_bounds, max_bounds) as numpy arrays of shape [3]
    """
    positions  = position.detach().cpu().numpy()
    min_bounds = positions.min(axis=0) - grid_config.padding
    max_bounds = positions.max(axis=0) + grid_config.padding
    return min_bounds, max_bounds


def create_coordinate_axes(
    labels : tuple[str, str, str] = ("X", "Y", "Z"),
    origin : tuple[float, float, float] = (0, 0, 0),
    scale  : float = 1.0,
) -> Axes:
    """
    Create coordinate axes for orientation reference.
    
    Creates XYZ coordinate axes for the visualization to provide spatial 
    reference and orientation. The axes are centered at the specified origin
    and scaled according to the scale parameter. This helps orient viewers
    in 3D space and provides a sense of scale to the visualization.
    
    Args:
        labels : Labels for each axis (X, Y, Z)
        origin : Origin point for the axes in 3D space
        scale  : Size scaling factor for the axes
        
    Returns:
        PyVista axes object configured for the coordinate reference
    """
    return pv.Axes(
        show_actor  = True,
        actor_scale = scale,
        x_label     = labels[0],
        y_label     = labels[1],
        z_label     = labels[2],
    )


def create_temperature_grid(
    environment : Any,
    position    : Tensor,
    grid_config : BaseModel,
) -> ImageData:
    """
    Create a uniform grid of temperature values from the environment.
    
    Samples the temperature field from the environment data source at regular
    grid points within a bounding box around the swarm. The resulting UniformGrid
    contains temperature data suitable for volume rendering or isosurface extraction.
    
    Args:
        environment : The simulation environment with thermal data source
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid resolution and padding
        
    Returns:
        PyVista UniformGrid with temperature scalar field data
    """
    min_bounds, max_bounds = compute_grid_bounds(position, grid_config)
    
    resolution = np.array(grid_config.temperature_resolution)
    grid       = pv.ImageData(
        dimensions = grid_config.temperature_resolution,
        spacing    = (max_bounds - min_bounds) / (resolution - 1),
        origin     = min_bounds
    )
    
    grid_tensor         = torch.from_numpy(grid.points).float()
    temps, _            = environment.data_source.query_thermal(grid_tensor)
    grid["temperature"] = temps.cpu().numpy().ravel()
    
    return grid


def create_wind_grid(
    simulation  : Any,
    position    : Tensor,
    grid_config : BaseModel,
) -> PolyData:
    """
    Create a grid of wind vectors from the environment data source.
    
    Samples the wind field from the environment at regular intervals within
    a bounding box around the swarm. The resulting PolyData contains points
    and vector data suitable for glyph-based visualization of the wind field.
    
    Args:
        simulation  : The simulation environment with wind data source
        position    : Agent positions tensor of shape [N, 3]
        grid_config : Configuration for grid resolution and padding
        
    Returns:
        PyVista PolyData with wind vector field data at each grid point
    """
    min_bounds, max_bounds = compute_grid_bounds(position, grid_config)
    resolution = grid_config.wind_resolution
    
    spacing_grid = pv.ImageData(
        dimensions = (resolution, resolution, resolution),
        spacing    = (max_bounds - min_bounds) / (resolution - 1),
        origin     = min_bounds,
    )
    
    wind_grid                  = pv.PolyData(spacing_grid.points)
    grid_tensor                = torch.from_numpy(wind_grid.points).float()
    wind_vectors               = simulation.data_source.query_wind(grid_tensor)
    wind_grid["wind_velocity"] = wind_vectors.cpu().numpy()
    
    return wind_grid
