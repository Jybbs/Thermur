"""
Type definitions for the Thermur CLI.

This module consolidates all type protocols, TypedDicts, and type aliases used
throughout the CLI, providing a single source of truth for type definitions.
"""
from __future__         import annotations
from config.cli.schemas import *
from typing             import Any, Literal, NamedTuple, Protocol, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch import Tensor


class CfgItem(NamedTuple):
    """
    Configuration parameter with override indicator.

    Represents a single configuration parameter with its hierarchical path,
    current value, and whether it was overridden from the default.
    """
    is_override : bool  # Whether this parameter was overridden
    path        : str   # Dot-separated parameter path (e.g., "model.lr")
    value       : Any   # Current parameter value


class FlockBatch(Protocol):
    """
    Protocol for PyTorch Geometric Batch objects containing flock data.

    Defines the expected structure of batched graph data from the
    demonstration dataset, ensuring type safety for attribute access.
    """
    action        : Tensor  # Expert actions            [B*N, 3]
    batch         : Tensor  # Node to graph assignment  [B*N]
    edge_index    : Tensor  # Graph edges               [2, E]
    gradient      : Tensor  # Temperature gradients     [B*N, 3]
    heterogeneity : Tensor  # Noise amplitudes η_i      [B*N]
    num_graphs    : int     # Number of graphs in batch
    position      : Tensor  # Agent positions           [B*N, 3]
    ptr           : Tensor  # Cumulative node counts    [B+1]
    temperature   : Tensor  # Temperature values        [B*N, 1]
    velocity      : Tensor  # Agent velocities          [B*N, 3]
    wind          : Tensor  # Wind velocities           [B*N, 3]
    x             : Tensor  # Node features             [B*N, 13]

    def __getitem__(self, key: str) -> Tensor:
        """
        Dictionary-style access to batch attributes.
        """
        ...


class SystemInfo(TypedDict, total=False):
    """
    System information dictionary returned by get_system_info.

    Contains comprehensive system details including hardware capabilities,
    software versions, and resource availability. Uses total=False to
    allow partial population based on available information.
    """
    # CUDA information
    cuda                : bool
    cuda_version        : str | None
    device_count        : int
    gpu_memory          : str | None
    gpu_name            : str | None

    # Dataset information
    dataset_count       : int
    dataset_size        : float
    has_sample          : bool

    # Resource information
    disk_available      : float
    disk_total          : float
    memory_available    : float
    memory_total        : float

    # Core versions
    platform            : str
    python              : str
    python_version_info : Any
    thermur             : str | None
    torch               : str


class TableColumn(NamedTuple):
    """
    Rich table column specification.

    Defines the structure for table column configuration used by Rich
    tables throughout the CLI for consistent styling and alignment.
    """
    justify : Literal["default", "left", "center", "right", "full"]
    style   : str
    title   : str
    width   : int
