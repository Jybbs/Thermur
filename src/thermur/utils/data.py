"""
Utilities for loading and querying environmental data.

This module provides a class to abstract away the details of reading and
interpolating from large, gridded datasets like the NetCDF files produced by
WRF-Fire.
"""
from configs.imitation import PhysicsModel, WRFDataModel
from numpy             import ndarray, zeros
from torch             import Tensor
from xarray            import DataArray, open_dataset

import torch


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
        data_path : str,
        physics   : PhysicsModel,
        wrf_data  : WRFDataModel
    ):
        """
        Loads the dataset from the specified path and initializes configuration.
        
        Args:
            data_path : Path to the NetCDF dataset file
            physics   : Physics configuration model with thermal interpolation settings
            wrf_data  : WRF data configuration for wind variables
        """
        self.dataset    = open_dataset(data_path, cache=True)
        self.coord_vars = list(self.dataset.coords)
        self.physics    = physics
        self.wrf_data   = wrf_data
        
        self.temp_var = physics.temperature_variable
        self.u_var    = self.wrf_data.u_wind_variable
        self.v_var    = self.wrf_data.v_wind_variable
        self.w_var    = self.wrf_data.w_wind_variable

    def _calculate_gradient(
        self, 
        coords_dict   : dict,
        positions     : Tensor,
        variable_name : str
    ) -> ndarray:
        """
        Calculates field gradient using finite differences.
        
        For each agent position, this method computes the 3D gradient ∇F 
        of a field variable F by sampling at offset positions and calculating 
        partial derivatives:
        
            ∂F/∂x ≈ (F(𝐱+εî) - F(𝐱-εî)) / 2ε
            ∂F/∂y ≈ (F(𝐱+εĵ) - F(𝐱-εĵ)) / 2ε
            ∂F/∂z ≈ (F(𝐱+εk̂) - F(𝐱-εk̂)) / 2ε
        
        Args:
            coords_dict   : Dictionary mapping dataset coordinates to position values
            positions     : Original position tensor [N, 3]
            variable_name : Name of the field variable to differentiate
            
        Returns:
            numpy.ndarray [N, 3] containing field gradients
        """
        epsilon         = self.physics.epsilon
        num_agents, dim = positions.shape
        gradients       = zeros((num_agents, dim))
        axes            = ['x', 'y', 'z']
        dim_names       = [
            self.physics.x_dimension, 
            self.physics.y_dimension, 
            self.physics.z_dimension
        ]
        
        for i in range(min(dim, len(axes))):
            dim_name = dim_names[i]
            if dim_name not in self.coord_vars:
                continue
            
            pos_values = self._interpolate_field(
                {**coords_dict, dim_name: coords_dict[dim_name] + epsilon},
                variable_name
            )
            neg_values = self._interpolate_field(
                {**coords_dict, dim_name: coords_dict[dim_name] - epsilon},
                variable_name
            )
            
            gradients[:, i] = (pos_values - neg_values) / (2 * epsilon)
        
        return gradients
    
    def _handle_out_of_bounds(
        self, 
        gradients    : Tensor,
        temperatures : Tensor 
    ) -> tuple[Tensor, Tensor]:
        """
        Processes NaN values resulting from out-of-bounds interpolation.
        
        This method identifies points where interpolation failed (typically because the
        agent position was outside the dataset domain) and replaces NaN values with
        sensible defaults to prevent numerical issues in downstream calculations.
        
        For out-of-bounds temperature values, we use a configurable fallback value.
        For gradients, we create a default upward-pointing gradient (in the 
        z-dimension), mimicking the behavior in ExpertFlockingController's
        vertical heat gradient method.
        
        Args:
            gradients    : Tensor [N, 3] of temperature gradients
            temperatures : Tensor [N, 1] of interpolated temperatures
            
        Returns:
            Tuple of processed (temperatures, gradients) with NaNs handled
        """
        nan_mask = torch.isnan(temperatures).squeeze()
        
        if not nan_mask.any():
            return temperatures, gradients
            
        fallback_temp = Tensor(
            [self.physics.fallback_temperature], 
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
        
    def _interpolate_field(
        self, 
        coords        : dict,
        variable_name : str
    ) -> ndarray:
        """
        Interpolates a field variable at the specified coordinates.
        
        For a batch of agent positions 𝐱₁, 𝐱₂, ..., 𝐱ₙ, this method interpolates
        the requested field at each position. This vectorized operation returns 
        exactly one value per position.
        
        Args:
            coords        : Dictionary mapping dimensions to coordinate arrays
            variable_name : Name of the NetCDF variable to interpolate
            
        Returns:
            Array of interpolated values at each position
        """
        n_points = len(next(iter(coords.values())))
        result   = [
            float(
                self.dataset[variable_name].interp(
                    coords = {d: a[i:i + 1] for d, a in coords.items()},
                    kwargs = {"fill_value": 0.0},
                    method = "linear"
                ).values
            )
            for i in range(n_points)
        ]
            
        return zeros(n_points) + result
        
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
            kwargs = {"fill_value": self.physics.fill_value},
            method = "linear"
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
            0 : self.physics.x_dimension,
            1 : self.physics.y_dimension,
            2 : self.physics.z_dimension
        }
        
        return {
            dim_mapping[i]: position_array[:, i]
            for i in range(min(3, position_array.shape[1]))
            if dim_mapping[i] in self.coord_vars
        }

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
        temp_values    = self._interpolate_field(coords_dict, self.temp_var)
        temp_gradients = self._calculate_gradient(coords_dict, positions, self.temp_var)
        
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
