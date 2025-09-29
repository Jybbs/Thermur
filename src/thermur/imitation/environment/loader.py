"""
WRF-Fire data loading and querying utilities.

This module provides a comprehensive data loader for WRF-Fire NetCDF outputs,
handling temperature, wind, and fire-specific variables with efficient
interpolation and gradient computation.
"""
from __future__          import annotations
from torch               import as_tensor, stack, tensor
from torch.func          import grad
from torch.nn.functional import grid_sample
from typing              import TYPE_CHECKING
from xarray              import open_dataset

if TYPE_CHECKING:
    from torch import Tensor


class WRFLoader:
    """
    Loader for WRF-Fire NetCDF datasets with efficient field queries.

    This class loads and caches WRF-Fire output files, providing vectorized
    methods to sample temperature, wind, and fire fields at arbitrary agent
    positions. It handles:
    - Coordinate transformations between simulation and dataset spaces
    - Temperature gradient computation using automatic differentiation
    - WRF perturbation temperature conversion: T = θ' + T₀ (T₀ = 300K)
    - Batched interpolation for all agents simultaneously

    WRF uses an Arakawa C-grid where velocity components are staggered:
    - U is defined at west/east cell faces (west_east_stag dimension)
    - V is defined at south/north cell faces (south_north_stag dimension)
    - W is defined at top/bottom cell faces (bottom_top_stag dimension)
    """
    def __init__(self, bounds_min: list[float], bounds_max: list[float]):
        """
        Initialize WRF data source for environmental field queries.

        Creates a data source that provides temperature and wind fields from
        WRF-SFIRE NetCDF outputs. Supports pinning to a specific snapshot for
        trajectory generation with consistent environmental conditions.

        Args:
            bounds_min : Lower bounds [x_min, y_min, z_min] for interpolation
            bounds_max : Upper bounds [x_max, y_max, z_max] for interpolation
        """
        self.bounds_max  = tensor(bounds_max)
        self.bounds_min  = tensor(bounds_min)
        self.n_snapshots = 0
        self.snapshot_idx = 0

    def _interpolate(self, field: Tensor, positions: Tensor) -> Tensor:
        """
        Interpolate field at agent positions using batched trilinear interpolation.

        Maps positions from simulation bounds to [-1, 1] for grid_sample, which
        expects normalized coordinates. PyTorch's grid_sample performs hardware-
        accelerated trilinear interpolation for all agents in a single CUDA kernel,
        replacing hundreds of sequential xarray.interp calls.

        Args:
            field     : Field tensor [C, Z, Y, X] to interpolate
            positions : Agent positions [N, 3] in simulation coordinates

        Returns:
            Interpolated values [N, C]
        """
        normalized = (
            2.0 * (positions - self.bounds_min) /
            (self.bounds_max - self.bounds_min) - 1.0
        )

        return grid_sample(
            align_corners = True,
            grid          = normalized.view(1, 1, 1, -1, 3),
            input         = field.unsqueeze(0) if field.dim() == 4 else field,
            mode          = 'bilinear',
            padding_mode  = 'zeros'
        ).squeeze((0, 2, 3)).T

    def load_datasets(self, file_paths: list[str]):
        """
        Load NetCDF datasets and pre-convert to tensors.

        Pre-loads WRF data as tensors for fast batched interpolation,
        eliminating per-frame xarray overhead.

        Args:
            file_paths: Absolute paths to NetCDF files from PyG's raw_paths
        """
        with open_dataset(
            drop_variables  = [
                'FGRNHFX', 'FXLAT', 'FXLONG', 'GRNHFX', 'P', 'PB',
                'PH', 'PHB', 'QVAPOR', 'Times', 'XTIME', 'tr17_1'
            ],
            engine          = 'netcdf4',
            filename_or_obj = file_paths[0]
        ) as ds:
            self.n_snapshots = ds.sizes['Time']
            self.temperature = as_tensor(ds['T'].values + 300.0)
            self.wind        = stack(
                [
                    as_tensor(ds['U'].values[..., :-1]),
                    as_tensor(ds['V'].values[..., :-1, :]),
                    as_tensor(ds['W'].values[:, :-1, ...])
                ], 
                dim = 1
            )

    def query_thermal(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Query temperature and its gradient at agent positions.

        Uses pre-loaded tensor data and batched interpolation for efficiency.
        Gradients are computed via automatic differentiation using nested
        lambdas, where p is the positions tensor and t is the interpolated
        temperature values.

        Args:
            positions: Tensor [N, 3] of agent positions in meters

        Returns:
            Tuple of (gradients, temperatures) where:
            - gradients    : Tensor [N, 3] of temperature gradients ∇T
            - temperatures : Tensor [N, 1] of temperature values in Kelvin
        """
        return grad(
            lambda p: (lambda t: (t.sum(), t))
            (
                self._interpolate(
                    self.temperature[self.snapshot_idx].unsqueeze(0), p
                )
            ),
            has_aux = True
        )(positions)

    def query_wind(self, positions: Tensor) -> Tensor:
        """
        Query wind velocity vectors at agent positions.

        Samples the WRF wind field 𝐮(𝐱) = [U, V, W] using batched interpolation.

        Args:
            positions: Tensor [N, 3] of agent positions in meters

        Returns:
            Tensor [N, 3] of wind velocity vectors 𝐮 = [u, v, w] in m/s
        """
        return self._interpolate(self.wind[self.snapshot_idx], positions)
