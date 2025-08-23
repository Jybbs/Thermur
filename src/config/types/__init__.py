"""
Type definitions for the Thermur CLI.

This module consolidates all type protocols, TypedDicts, and type aliases used
throughout the CLI, providing a single source of truth for type definitions.
"""
from __future__         import annotations
from config.cli.schemas import *
from typing             import Any, Literal, NamedTuple, TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from torch import Tensor


class ConfigItem(NamedTuple):
    """
    Configuration parameter with override indicator.

    Represents a single configuration parameter with its hierarchical path,
    current value, and whether it was overridden from the default.
    """
    is_override : bool  # Whether this parameter was overridden
    path        : str   # Dot-separated parameter path (e.g., "model.lr")
    value       : Any   # Current parameter value


class StepMetrics(TypedDict):
    """
    Step-level metrics data for logging during training/validation.

    Contains the loss value and model predictions/targets needed for
    computing step-level metrics like per-dimension MSE. When None,
    indicates epoch-level aggregated logging only.
    """
    loss        : Tensor  # Scalar loss value
    predictions : Tensor  # Model velocity predictions [batch, 3]
    targets     : Tensor  # Expert velocity targets [batch, 3]


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


