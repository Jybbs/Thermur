"""
Utilities for loading and querying environmental data.

This module provides a class to abstract away the details of reading and
interpolating from large, gridded datasets like the NetCDF files produced by
WRF-Fire.
"""
from torch  import Tensor
from xarray import open_dataset


class EnvironmentDataSource:
    """
    A wrapper for environmental data providing efficient queries.
    """
    def __init__(self, data_path: str):
        """
        Loads the dataset from the specified path.
        """
        self.dataset = open_dataset(data_path, cache=True)

    def query_thermal(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        """
        Queries temperature and its gradient for a batch of positions.
        """
        raise NotImplementedError("Data interpolation logic to be implemented.")
