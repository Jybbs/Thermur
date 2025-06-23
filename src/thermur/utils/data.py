"""
Utilities for loading and querying environmental data.

This module provides a class to abstract away the details of reading and
interpolating from large, gridded datasets like the NetCDF files produced by
WRF-Fire.
"""
import torch

from numpy  import ndarray, zeros
from torch  import Tensor
from xarray import DataArray, open_dataset


class EnvironmentDataSource:
    """
    A wrapper for environmental data providing efficient queries.
    
    This class loads and caches NetCDF datasets, providing vectorized methods to
    sample the continuous temperature field at arbitrary agent positions. It handles
    coordinate transformations between the simulation space and the dataset's
    coordinate system, and calculates temperature gradients using finite differences.
    """
    def __init__(
        self, 
        data_path: str, 
        interpolation = None
    ):
        """
        Loads the dataset from the specified path and initializes configuration.
        
        Args:
            data_path     : Path to the NetCDF dataset file
            interpolation : Configuration parameters for thermal data interpolation
        """
        self.dataset    = open_dataset(data_path, cache=True)
        self.config     = interpolation
        self.coord_vars = list(self.dataset.coords)
        
        if hasattr(self.config, 'temperature_variable'):
            self.temp_var = self.config.temperature_variable

        else:
            temp_vars = [v for v in self.dataset.variables 
                        if any(name in v.lower() for name in ['t', 'temp'])]
            
            self.temp_var = temp_vars[0] if temp_vars else next(
                (v for v in self.dataset.variables if v not in self.coord_vars), 
                None
            )

    def query_thermal(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Queries temperature and its gradient for a batch of positions.
        
        This method efficiently samples the gridded NetCDF dataset to retrieve 
        temperature values at arbitrary points in 3D space. It performs a vectorized 
        interpolation for the entire batch of agent positions and calculates the 
        temperature gradient using finite differences.
        
        The workflow involves transforming coordinates, interpolating temperature
        values, calculating gradients using central differences, and handling
        out-of-bounds positions with NaN indicators.
        
        Args:
            positions: Tensor [N, 3] containing N agent positions in simulation 
                       coordinates
                
        Returns:
            A tuple of (temperature, gradient) where:
            - temperature : Tensor [N, 1] of interpolated temperature values
            - gradient    : Tensor [N, 3] of temperature gradients (∇T)
        """
        coords_dict    = self._transform_coordinates(positions)
        temp_values    = self._interpolate_along_dimension(coords_dict)
        temp_gradients = self._calculate_gradient(positions, coords_dict)
        
        return self._handle_out_of_bounds(
            Tensor(
                temp_values.reshape(-1, 1), 
                device = positions.device
            ),
            Tensor(
                temp_gradients, 
                device = positions.device
            )
        )

    def _calculate_gradient(
        self, 
        positions   : Tensor, 
        coords_dict : dict
    ) -> ndarray:
        """
        Calculates temperature gradient using finite differences.
        
        For each agent position, this method computes the 3D temperature 
        gradient ∇T by sampling temperature at offset positions and 
        calculating partial derivatives:
        
            ∂T/∂x ≈ (T(𝐱+εî) - T(𝐱-εî)) / 2ε
            ∂T/∂y ≈ (T(𝐱+εĵ) - T(𝐱-εĵ)) / 2ε
            ∂T/∂z ≈ (T(𝐱+εk̂) - T(𝐱-εk̂)) / 2ε
        
        Args:
            positions   : Original position tensor [N, 3]
            coords_dict : Dictionary mapping dataset coordinates to position values
            
        Returns:
            numpy.ndarray [N, 3] containing temperature gradients
        """
        epsilon         = self.config.epsilon
        num_agents, dim = positions.shape
        gradients       = zeros((num_agents, dim))
        axes            = ['x', 'y', 'z']
        
        for i in range(min(dim, len(axes))):
            dim_name = getattr(self.config, f"{axes[i]}_dimension")
            if dim_name not in self.coord_vars:
                continue
            
            pos_temps = self._interpolate_along_dimension(
                {**coords_dict, dim_name: coords_dict[dim_name] + epsilon}
            )
            neg_temps = self._interpolate_along_dimension(
                {**coords_dict, dim_name: coords_dict[dim_name] - epsilon}
            )
            
            gradients[:, i] = (pos_temps - neg_temps) / (2 * epsilon)
        
        return gradients
        
    def _interpolate_along_dimension(self, coords: dict) -> ndarray:
        """
        Samples temperature at the specified coordinates.
        
        For a batch of agent positions 𝐱₁, 𝐱₂, ..., 𝐱ₙ, this method interpolates
        the temperature field T at each position. This vectorized operation 
        returns exactly one temperature value per position.
        
        Args:
            coords : Dictionary mapping dimensions (`d`) to coordinate arrays (`a`)
            
        Returns:
            Array of temperature values T(𝐱₁), T(𝐱₂), ..., T(𝐱ₙ)
        """
        n_points = len(next(iter(coords.values())))
        result   = [
            float(self.dataset[self.temp_var].interp(
                coords = {d: a[i:i + 1] for d, a in coords.items()},
                method = "linear",
                kwargs = {"fill_value": self.config.fill_value},
            ).values)
            for i in range(n_points)
        ]
            
        return zeros(n_points) + result

    def _handle_out_of_bounds(
        self, 
        temperatures : Tensor, 
        gradients    : Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Processes NaN values resulting from out-of-bounds interpolation.
        
        This method identifies points where interpolation failed (typically because the
        agent position was outside the dataset domain) and replaces NaN values with
        sensible defaults to prevent numerical issues in downstream calculations.
        
        For out-of-bounds temperature values, we use a configurable fallback value.
        For gradients, we create a default upward-pointing gradient (in the z-dimension),
        mimicking the behavior in ExpertFlockingController._vertical_heat_gradient.
        
        Args:
            temperatures : Tensor [N, 1] of interpolated temperatures
            gradients    : Tensor [N, 3] of temperature gradients
            
        Returns:
            tuple of processed (temperatures, gradients) with NaNs handled
        """
        nan_mask = torch.isnan(temperatures).squeeze()
        
        if not nan_mask.any():
            return temperatures, gradients
            
        fallback_temp = Tensor(
            [self.config.fallback_temperature], 
            device = temperatures.device
        )
        
        default_grad        = torch.zeros_like(gradients)
        default_grad[:, -1] = 1.0
        
        return (
            torch.where(
                condition = torch.isnan(temperatures), 
                input     = fallback_temp.expand_as(temperatures), 
                other     = temperatures
            ),
            torch.where(
                condition = torch.isnan(gradients), 
                input     = default_grad, 
                other     = gradients
            )
        )
        
    def _interpolate_temperature(self, coords_dict: dict) -> DataArray:
        """
        Performs vectorized interpolation of temperature data.
        
        This method uses xarray's efficient interpolation capabilities to sample
        temperature values for an entire batch of agent positions in a single
        operation, avoiding Python loops over individual agents.
        
        Args:
            coords_dict: Dictionary mapping dataset coordinates to position values
            
        Returns:
            xarray.DataArray containing interpolated temperatures
        """
        return self.dataset[self.temp_var].interp(
            coords = coords_dict, 
            method = "linear", 
            kwargs = {"fill_value": self.config.fill_value}
        )
    
    def _transform_coordinates(self, positions: Tensor) -> dict[str, ndarray]:
        """
        Transforms simulation coordinates to dataset coordinates.
        
        This method handles the mapping between the continuous coordinate system
        of the simulation and the potentially different coordinate system used
        in the gridded dataset.
        
        Args:
            positions: Tensor [N, 3] of agent positions
            
        Returns:
            Dictionary mapping dataset coordinate names to position values
        """
        position_array = positions.detach().cpu().numpy()
        dim_mapping    = {
            0 : self.config.x_dimension,
            1 : self.config.y_dimension,
            2 : self.config.z_dimension
        }
        
        return {
            dim_mapping[i]: position_array[:, i]
            for i in range(min(3, position_array.shape[1]))
            if dim_mapping[i] in self.coord_vars
        }
