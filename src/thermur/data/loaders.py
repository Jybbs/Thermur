"""
WRF-Fire data loading and querying utilities.

This module provides a comprehensive data loader for WRF-Fire NetCDF outputs,
handling temperature, wind, and fire-specific variables with efficient 
interpolation and gradient computation.
"""
from configs.imitation import PhysicsModel, WRFDataModel
from numpy             import ndarray, zeros
from torch             import Tensor
from xarray            import open_dataset

import torch


class WRFDataSource:
    """
    Loader for WRF-Fire NetCDF datasets with efficient field queries.
    
    This class loads and caches WRF-Fire output files, providing vectorized 
    methods to sample temperature, wind, and fire fields at arbitrary agent 
    positions. It handles:
    
    - Coordinate transformations between simulation and dataset spaces
    - Temperature gradient computation using finite differences
    - WRF perturbation temperature conversion (T + 300K)
    - Staggered grid interpolation for wind components
    - Fire-specific variables like ground heat flux
    
    WRF uses an Arakawa C-grid where velocity components are staggered:
    - U is defined at west/east cell faces (west_east_stag dimension)
    - V is defined at south/north cell faces (south_north_stag dimension)  
    - W is defined at top/bottom cell faces (bottom_top_stag dimension)
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
            data_path : Path to the WRF-Fire NetCDF dataset file
            physics   : Physics configuration model with interpolation settings
            wrf_data  : WRF data configuration for variable names
        """
        self.dataset    = open_dataset(data_path, cache=True)
        self.coord_vars = list(self.dataset.coords)
        self.physics    = physics
        self.wrf_data   = wrf_data
        
        self.temp_var = physics.temperature_variable
        self.u_var    = wrf_data.u_wind_variable
        self.v_var    = wrf_data.v_wind_variable
        self.w_var    = wrf_data.w_wind_variable

    def _add_domain_noise(self, data: Tensor, noise_std: float) -> Tensor:
        """
        Add domain randomization noise if enabled.
        
        Args:
            data      : Tensor to add noise to
            noise_std : Standard deviation of Gaussian noise
            
        Returns:
            Tensor with noise added if domain randomization is enabled
        """
        if self.wrf_data.domain_randomization and noise_std > 0:
            return data + torch.randn_like(data) * noise_std
        
        return data

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
        dim_names       = [
            self.physics.x_dimension, 
            self.physics.y_dimension, 
            self.physics.z_dimension
        ]
        
        for i in range(min(dim, 3)):
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
        
        This method identifies points where interpolation failed (typically because 
        the agent position was outside the dataset domain) and replaces NaN values 
        with sensible defaults to prevent numerical issues in downstream calculations.
        
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
        nan_grad_mask       = torch.isnan(gradients).any(dim=1)
        default_grad        = torch.zeros_like(gradients)
        default_grad[:, -1] = 1.0
        
        gradients[nan_grad_mask] = default_grad[nan_grad_mask]
        
        return (
            torch.nan_to_num(temperatures, nan=self.physics.fallback_temperature),
            gradients
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
        return self.dataset[variable_name].interp(
            coords = coords,
            kwargs = {"fill_value": 0.0},
            method = "linear"
        ).values.astype(float)
    
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

    def get_domain_info(self) -> dict:
        """
        Extract WRF domain configuration information.
        
        Returns:
            Dictionary containing domain metadata like grid spacing,
            projection information, and simulation time
        """
        attrs = self.dataset.attrs
        
        return {
            "cen_lat"     : attrs.get("CEN_LAT", None),
            "cen_lon"     : attrs.get("CEN_LON", None),
            "dt"          : attrs.get("DT", None),
            "dx"          : attrs.get("DX", None),
            "dy"          : attrs.get("DY", None),
            "grid_id"     : attrs.get("GRID_ID", None),
            "map_proj"    : attrs.get("MAP_PROJ", None),
            "parent_id"   : attrs.get("PARENT_ID", None),
            "start_date"  : attrs.get("START_DATE", None)
        }
    
    def query_fire_heat_flux(self, positions: Tensor) -> Tensor:
        """
        Query ground heat flux from fire.
        
        GRNHFX represents the sensible heat flux at the surface due
        to fire, measured in W/m². This is a 2D field, so the query
        ignores the vertical coordinate.
        
        Args:
            positions: Tensor [N, 3] of agent positions
            
        Returns:
            Tensor [N, 1] of heat flux values in W/m²
        """
        coords_dict = self._transform_coordinates(positions)
        coords_2d   = {k: v for k, v in coords_dict.items() if k != self.physics.z_dimension}
        
        heat_flux = self._interpolate_field(coords_2d, self.wrf_data.fire_heat_variable)
        return torch.nan_to_num(
            Tensor(heat_flux.reshape(-1, 1), device=positions.device),
            nan = 0.0
        )

    def query_thermal(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Queries temperature and its gradient for a batch of positions.
        
        This method efficiently samples the gridded NetCDF dataset to retrieve 
        temperature values at arbitrary points in 3D space. It performs a vectorized 
        interpolation for the entire batch of agent positions and calculates the 
        temperature gradient using finite differences.
        
        WRF stores temperature as perturbation potential temperature, where
        actual temperature = T + 300K. This method handles the conversion
        automatically when the temperature variable is "T".
        
        Domain randomization adds Gaussian noise to improve robustness during
        training, simulating sensor noise and environmental uncertainty.
        
        Args:
            positions: Tensor [N, 3] containing N agent positions in simulation 
                       coordinates
                
        Returns:
            A tuple of (temperature, gradient) where:
            - temperature : Tensor [N, 1] of interpolated temperature values
            - gradient    : Tensor [N, 3] of temperature gradients (∇T)
        """
        coords_dict = self._transform_coordinates(positions)
        
        temp_values = Tensor(
            self._interpolate_field(coords_dict, self.temp_var).reshape(-1, 1),
            device = positions.device
        )
        temp_gradients = Tensor(
            self._calculate_gradient(coords_dict, positions, self.temp_var),
            device = positions.device
        )
        
        temperatures, gradients = self._handle_out_of_bounds(temp_gradients, temp_values)
        
        if self.temp_var == "T":
            temperatures = temperatures + 300.0
        
        temperatures = self._add_domain_noise(
            temperatures, 
            self.wrf_data.temperature_noise_std
        )
        
        return temperatures, gradients
    
    def query_wind(self, positions: Tensor) -> Tensor:
        """
        Queries wind velocity vectors for a batch of positions.
        
        This method samples the U, V, W wind components from the WRF-Fire
        dataset at arbitrary agent positions. WRF uses a staggered grid where
        wind components are defined at cell faces rather than centers, but
        this method handles the interpolation transparently.
        
        If wind data is not available in the dataset, returns zero vectors
        to maintain compatibility with simulations that don't include wind.
        
        Domain randomization adds Gaussian noise to each wind component to
        simulate measurement uncertainty and turbulent fluctuations.
        
        Args:
            positions: Tensor [N, 3] containing N agent positions in simulation
                       coordinates
                       
        Returns:
            Tensor [N, 3] of wind velocity vectors [u, v, w] in m/s
        """
        coords_dict = self._transform_coordinates(positions)
        wind_values = torch.stack(
            dim     = 1,
            tensors = [
                Tensor(
                    self._interpolate_field(coords_dict, v), 
                    device = positions.device
                ) for v in [self.u_var, self.v_var, self.w_var]
            ]
        )
        
        wind_values = torch.nan_to_num(wind_values, 0.0)

        return self._add_domain_noise(wind_values, self.wrf_data.wind_noise_std)
