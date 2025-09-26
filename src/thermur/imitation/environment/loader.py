"""
WRF-Fire data loading and querying utilities.

This module provides a comprehensive data loader for WRF-Fire NetCDF outputs,
handling temperature, wind, and fire-specific variables with efficient
interpolation and gradient computation.
"""
from __future__ import annotations
from numpy      import zeros
from torch      import Tensor
from typing     import Any, Mapping, TYPE_CHECKING
from xarray     import DataArray, open_dataset

import torch as th

if TYPE_CHECKING:
    from config.imitation.environment import LoaderModel, PhysicsModel
    from numpy.typing                 import NDArray


class WRFLoader:
    """
    Loader for WRF-Fire NetCDF datasets with efficient field queries.

    This class loads and caches WRF-Fire output files, providing vectorized
    methods to sample temperature, wind, and fire fields at arbitrary agent
    positions. It handles:

    - Coordinate transformations between simulation and dataset spaces
    - Temperature gradient computation using finite differences
    - WRF perturbation temperature conversion: T = θ' + T₀ (T₀ = 300K)
    - Staggered grid interpolation for wind components
    - Fire heat flux querying (not yet integrated into training)

    WRF uses an Arakawa C-grid where velocity components are staggered:
    - U is defined at west/east cell faces (west_east_stag dimension)
    - V is defined at south/north cell faces (south_north_stag dimension)
    - W is defined at top/bottom cell faces (bottom_top_stag dimension)
    """
    def __init__(
        self,
        loader  : LoaderModel,
        physics : PhysicsModel
    ):
        """
        Initialize WRF data source for environmental field queries.

        Creates a data source that provides temperature, wind, and fire heat flux
        fields from WRF-SFIRE NetCDF outputs. Automatically discovers all valid
        NetCDF files in the data directory. Handles temporal interpolation between
        snapshots and stochastic domain randomization for robust training.

        Args:
            loader  : Data loading configuration including noise parameters
            physics : Physics simulation parameters for gradient computation
        """
        self.bounds_max            = th.tensor(physics.bounds_max)
        self.bounds_min            = th.tensor(physics.bounds_min)
        self.domain_noise_rng      = th.Generator()
        self.domain_randomization  = loader.domain_randomization
        self.fallback_temperature  = physics.fallback_temperature
        self.interpolate_time      = loader.interpolate_time
        self.temperature_noise_std = loader.temperature_noise_std
        self.wind_noise_std        = loader.wind_noise_std

    def _add_domain_noise(self, data: Tensor, noise_std: float, timestep: int) -> Tensor:
        """
        Add domain randomization noise if enabled.

        Args:
            data      : Tensor to add noise to
            noise_std : Standard deviation of Gaussian noise
            timestep  : Current simulation timestep for seeding

        Returns:
            Tensor with noise added if domain randomization is enabled
        """
        self.domain_noise_rng.manual_seed(timestep)
        return (
            data + th.randn(data.shape, generator=self.domain_noise_rng) * noise_std
            if self.domain_randomization and noise_std > 0
            else data
        )

    def _calculate_gradient(
        self,
        coords_dict   : Mapping[str, Tensor | NDArray[Any]],
        positions     : Tensor,
        variable_name : str,
        epsilon       : float = 1e-3
    ) -> NDArray[Any]:
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
            epsilon       : Step size for finite differences (default 1e-3 for
                            numerical stability)

        Returns:
            numpy.ndarray [N, 3] containing field gradients
        """
        dim_names = ["west_east", "south_north", "bottom_top"]
        gradients = zeros((positions.shape[0], 3))

        for i, dim_name in enumerate(dim_names):
            if dim_name not in self.coord_vars:
                continue

            gradients[:, i] = (
                self._interpolate_field(
                    {**coords_dict, dim_name: coords_dict[dim_name] + epsilon},
                    variable_name
                ) -
                self._interpolate_field(
                    {**coords_dict, dim_name: coords_dict[dim_name] - epsilon},
                    variable_name
                )
            ) / (2 * epsilon)

        return gradients

    def _extract_dataset_metadata(self):
        """
        Extract metadata from the current dataset for coordinate transformations.

        Sets up coordinate variables, time steps, and grid dimensions needed
        for field interpolation and temporal navigation.
        """
        self.coord_vars    = list(self.dataset.dims)
        self.num_timesteps = self.dataset.sizes.get('Time', 1)
        self.time_interval = 300.0

        self.grid_dims = {
            "west_east"   : self.dataset.sizes.get("west_east",   500),
            "south_north" : self.dataset.sizes.get("south_north", 250),
            "bottom_top"  : self.dataset.sizes.get("bottom_top",  50)
        }

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
        if (nan_mask := th.isnan(gradients).any(dim=1)).any():
            default_grad        = th.zeros_like(gradients)
            default_grad[:, -1] = 1.0
            gradients[nan_mask] = default_grad[nan_mask]

        return (
            gradients,
            th.nan_to_num(temperatures, self.fallback_temperature)
        )

    def _get_time_indices(self) -> tuple[int, int, float]:
        """
        Map continuous time to discrete WRF snapshot indices.

        Given simulation time t ∈ ℝ⁺, computes indices (i₀, i₁) and weight α
        for temporal interpolation between snapshots:

            t_continuous = (t + t_offset) / Δt_wrf
            i₀ = ⌊t_continuous⌋ mod N
            i₁ = (i₀ + 1) mod N
            α = t_continuous - i₀

        where N is the number of time steps and Δt_wrf is the WRF output interval.

        Returns:
            Tuple (i₀, i₁, α) for linear time interpolation
        """
        continuous_idx = (self.time / self.time_interval) % self.num_timesteps

        return (
            (lower := int(continuous_idx)),
            (lower + 1) % self.num_timesteps,
            continuous_idx - lower
        )

    def _interpolate_field(
        self,
        coords        : Mapping[str, Tensor | NDArray[Any]],
        variable_name : str
    ) -> NDArray[Any]:
        """
        Interpolate field variable at agent positions using trilinear interpolation.

        For a field F defined on a regular grid, computes F(𝐱ᵢ) for each agent
        position 𝐱ᵢ ∈ ℝ³. When temporal interpolation is enabled, performs
        4D interpolation:

            F(𝐱, t) = (1 - α) · F(𝐱, t₀) + α · F(𝐱, t₁)

        where t₀ and t₁ are adjacent time snapshots and α ∈ [0, 1] is the
        temporal blending weight. Spatial interpolation uses trilinear basis
        functions on the WRF Arakawa C-grid.

        Args:
            coords        : Coordinate arrays {dim: positions} for N agents
            variable_name : NetCDF variable to interpolate

        Returns:
            Array [N] of interpolated field values
        """
        interp = lambda d: d.interp(
            **{dim: DataArray(v, dims='agent') for dim, v in coords.items()},
            kwargs = {"fill_value": 0.0},
            method = "linear"
        ).values.astype(float)

        data = self.dataset[variable_name]

        if "Time" not in data.dims:
            return interp(data)

        if not self.interpolate_time or self.num_timesteps <= 1:
            return interp(
                data.isel(
                    Time = int(self.time / self.time_interval) % self.num_timesteps
                )
            )

        lower_idx, upper_idx, weight = self._get_time_indices()

        return (
            interp(data.isel(Time=lower_idx)) * (1 - weight) +
            interp(data.isel(Time=upper_idx)) * weight
        )

    def _transform_coordinates(self, positions: Tensor) -> dict[str, Tensor]:
        """
        Transform simulation coordinates to WRF grid indices.

        Maps agent positions 𝐱 ∈ [𝐱_min, 𝐱_max]³ from the simulation's physical
        coordinate system to WRF grid indices 𝐢 ∈ [0, N-1]³ via linear interpolation:

            𝐢 = (𝐱 - 𝐱_min) / (𝐱_max - 𝐱_min) · (N - 1)

        where N represents the grid dimensions for each axis. This ensures agents
        operating within the physical workspace bounds are properly mapped to the
        WRF computational domain for field queries.

        Args:
            positions: Tensor [N, 3] of agent positions in meters

        Returns:
            Dictionary mapping WRF dimension names to grid indices as tensors
        """
        device     = positions.device
        normalized = (
            (positions - self.bounds_min.to(device)) /
            (self.bounds_max.to(device) - self.bounds_min.to(device))
        )

        return {
            "west_east"   : normalized[:, 0] * (self.grid_dims["west_east"] - 1),
            "south_north" : normalized[:, 1] * (self.grid_dims["south_north"] - 1),
            "bottom_top"  : normalized[:, 2] * (self.grid_dims["bottom_top"] - 1)
        }

    def load_datasets(self, file_paths: list[str]):
        """
        Load NetCDF datasets from PyG-provided paths.

        Called by DemonstrationsDataset after PyG confirms data exists,
        either from disk or after automatic download.

        Args:
            file_paths: Absolute paths to NetCDF files from PyG's raw_paths
        """
        self.datasets = [
            open_dataset(
                cache           = True,
                engine          = 'netcdf4',
                filename_or_obj = path
            )
            for path in file_paths
        ]

        if self.datasets:
            self.dataset = self.datasets[0]
            self.time    = 0.0
            self._extract_dataset_metadata()

    def query_fire_heat_flux(self, positions: Tensor) -> Tensor:
        """
        Query ground heat flux from fire.

        TODO: Integrate fire heat flux into trajectory generation and training.
        This method extracts GRNHFX (ground heat flux from fire) from WRF-SFIRE
        simulations but is not yet integrated into the training pipeline.

        Future integration possibilities:
        - Add to PyG Data objects in TrajectoryGenerator.reset() and step()
        - Use in MurmurationController for fire avoidance behaviors:
          * Repulsive forces from high heat flux areas
          * Alert state triggering when fire danger detected
          * Modified flocking cohesion near fire zones
        - Incorporate into ThermalPenalty as soft constraints:
          * Thermal gradient corrections
          * Dynamic safety responses based on heat intensity
          * Smooth penalty escalation near danger zones

        This would enable agents to learn fire-aware (instead of just
        heat-aware) navigation strategies, which could better encapsulate
        how real flocks would avoid threatening smoke plumes and flare-like
        heat spikes.

        Args:
            positions: Tensor [N, 3] of agent positions

        Returns:
            Tensor [N, 1] of heat flux values in W/m²
        """
        coords_dict = self._transform_coordinates(positions)
        coords_2d   = {k: v for k, v in coords_dict.items() if k != "bottom_top"}

        heat_flux = self._interpolate_field(coords_2d, "GRNHFX")
        return th.nan_to_num(
            Tensor(heat_flux.reshape(-1, 1), device=positions.device),
            nan = 0.0
        )

    def query_thermal(
        self, 
        positions : Tensor, 
        timestep  : int
    ) -> tuple[Tensor, Tensor]:
        """
        Query temperature and its gradient at agent positions.

        Samples the WRF temperature field and computes its gradient using finite
        differences. WRF stores perturbation potential temperature θ' where:

            T(𝐱) = θ'(𝐱) + T₀

        with reference temperature T₀ = 300K. The gradient is computed as:

            ∇T ≈ [∂T/∂x, ∂T/∂y, ∂T/∂z]

        Domain randomization adds Gaussian noise ε ~ N(0, σ_T²) to simulate
        sensor uncertainty and atmospheric turbulence.

        Args:
            positions : Tensor [N, 3] of agent positions in meters
            timestep  : Current simulation timestep for deterministic noise

        Returns:
            Tuple of (gradient, temperature) where:
            - gradient    : Tensor [N, 3] of temperature gradients ∇T
            - temperature : Tensor [N, 1] of temperature values in Kelvin
        """
        coords_dict = self._transform_coordinates(positions)
        in_bounds   = self._handle_out_of_bounds(
            gradients = Tensor(
                self._calculate_gradient(coords_dict, positions, "T"),
                device = positions.device
            ),
            temperatures = Tensor(
                self._interpolate_field(coords_dict, "T").reshape(-1, 1),
                device = positions.device
            ) + 300.0
        )

        return (
            in_bounds[0],
            self._add_domain_noise(in_bounds[1], self.temperature_noise_std, timestep),
        )

    def query_wind(self, positions: Tensor, timestep: int) -> Tensor:
        """
        Query wind velocity vectors at agent positions.

        Samples the WRF wind field 𝐮(𝐱) = [U, V, W] from staggered grid
        components. On the Arakawa C-grid, velocity components are defined at
        cell faces:

            U: west_east_stag   (x-wind component)
            V: south_north_stag (y-wind component)
            W: bottom_top_stag  (z-wind component)

        Trilinear interpolation reconstructs velocities at arbitrary positions.
        Domain randomization adds noise ε ~ N(0, σ_w²) to simulate turbulence.

        Args:
            positions : Tensor [N, 3] of agent positions in meters
            timestep  : Current simulation timestep for deterministic noise

        Returns:
            Tensor [N, 3] of wind velocity vectors 𝐮 = [u, v, w] in m/s
        """
        base_coords = self._transform_coordinates(positions)
        device      = positions.device
        n           = positions.shape[0]

        # Map each wind component to its staggered grid (U→x, V→y, W→z)
        dims      = ["west_east", "south_north", "bottom_top"]
        staggered = {
            var: {
                (dims[i] + "_stag" if i == idx else dims[i]): base_coords[dims[i]]
                for i in range(3)
            }
            for idx, var in enumerate(["U", "V", "W"])
        }

        wind_values = th.stack([
            th.nan_to_num(
                Tensor(
                    self._interpolate_field(staggered[var], var),
                    device = device
                ),
                nan = 0.0
            ) if var in self.dataset.data_vars else th.zeros(n, device=device)
            for var in ["U", "V", "W"]
        ], dim=1)

        return self._add_domain_noise(wind_values, self.wind_noise_std, timestep)
