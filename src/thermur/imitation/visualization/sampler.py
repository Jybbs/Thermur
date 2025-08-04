"""
Spatial sampling utilities for visualization data sources.

This module provides methods for sampling and discretizing continuous simulation
data from the environment for visualization purposes. It creates grid-based
representations of thermal fields, wind vectors, and other spatially-distributed
data needed for 3D rendering.

The sampling functions efficiently handle large-scale data by using vectorized
operations and leveraging PyVista's optimized data structures.
"""
from __future__   import annotations
from numpy        import array
from numpy.typing import NDArray
from pyvista      import Axes, ImageData, PolyData
from torch        import from_numpy, Tensor
from typing       import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation.environment import SimulationEnv


class Sampler:
    """
    Manages spatial sampling of simulation data for visualization.

    This class provides methods for creating grid-based representations of
    continuous simulation data. It handles the discretization of thermal fields,
    wind vectors, and other spatially-distributed data into formats suitable
    for 3D rendering with PyVista.

    The sampler uses efficient vectorized operations to handle large-scale
    simulations while maintaining performance.
    """

    def __init__(
        self,
        grid_padding           : float,
        temperature_resolution : tuple[int, int, int],
        wind_resolution        : int
    ):
        """
        Initialize the grid sampler with configuration.

        Args:
            grid_padding           : Buffer distance for grid generation
            temperature_resolution : Voxel grid dimensions for temperature field
            wind_resolution        : Grid points per dimension for wind vectors
        """
        self.grid_padding           = grid_padding
        self.temperature_resolution = temperature_resolution
        self.wind_resolution        = wind_resolution

    def compute_grid_bounds(
        self,
        position : Tensor
    ) -> tuple[NDArray[Any], NDArray[Any]]:
        """
        Compute the bounding box for a grid based on agent positions.

        Calculates the minimum and maximum coordinates for a bounding box around
        the agent positions, with padding to ensure the entire simulation
        domain of interest is captured for visualization.

        Args:
            position: Agent positions tensor of shape [N, 3]

        Returns:
            Tuple of (min_bounds, max_bounds) as numpy arrays of shape [3]
        """
        positions  = position.detach().cpu().numpy()
        min_bounds = positions.min(axis=0) - self.grid_padding
        max_bounds = positions.max(axis=0) + self.grid_padding

        return min_bounds, max_bounds

    def create_coordinate_axes(
        self,
        scale : float = 1.0
    ) -> Axes:
        """
        Create coordinate axes for orientation reference.

        Creates XYZ coordinate axes for the visualization to provide spatial
        reference and orientation. The axes are scaled according to the scale
        parameter. This helps orient viewers in 3D space and provides a sense
        of scale to the visualization.

        Args:
            scale : Size scaling factor for the axes

        Returns:
            PyVista axes object configured for the coordinate reference
        """
        return Axes(
            actor_scale = int(scale),
            show_actor  = True
        )

    def create_temperature_grid(
        self,
        environment : SimulationEnv,
        position    : Tensor
    ) -> ImageData:
        """
        Create a uniform grid of temperature values from the environment.

        Samples the temperature field from the environment data source at regular
        grid points within a bounding box around the flock. The resulting UniformGrid
        contains temperature data suitable for volume rendering or isosurface extraction.

        Args:
            environment : The simulation environment with thermal data source
            position    : Agent positions tensor of shape [N, 3]

        Returns:
            PyVista UniformGrid with temperature scalar field data
        """
        min_bounds, max_bounds = self.compute_grid_bounds(position)

        resolution = array(self.temperature_resolution)
        grid       = ImageData(
            dimensions = self.temperature_resolution,
            origin     = min_bounds,
            spacing    = (max_bounds - min_bounds) / (resolution - 1)
        )

        grid_tensor         = from_numpy(grid.points).float()
        temps, _            = environment.wrf.query_thermal(grid_tensor)
        grid["temperature"] = temps.cpu().numpy().ravel()

        return grid

    def create_wind_grid(
        self,
        position   : Tensor,
        simulation : SimulationEnv
    ) -> PolyData:
        """
        Create a grid of wind vectors from the environment data source.

        Samples the wind field from the environment at regular intervals within
        a bounding box around the flock. The resulting PolyData contains points
        and vector data suitable for glyph-based visualization of the wind field.

        Args:
            position   : Agent positions tensor of shape [N, 3]
            simulation : The simulation environment with wind data source

        Returns:
            PyVista PolyData with wind vector field data at each grid point
        """
        min_bounds, max_bounds = self.compute_grid_bounds(position)
        resolution = self.wind_resolution

        spacing_grid = ImageData(
            dimensions = (resolution, resolution, resolution),
            origin     = min_bounds,
            spacing    = (max_bounds - min_bounds) / (resolution - 1)
        )

        wind_grid                  = PolyData(spacing_grid.points)
        grid_tensor                = from_numpy(wind_grid.points).float()
        wind_vectors               = simulation.wrf.query_wind(grid_tensor)
        wind_grid["wind_velocity"] = wind_vectors.cpu().numpy()

        return wind_grid
